from __future__ import annotations

import argparse
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd

from a_share_crawler import DEFAULT_KTYPE, mysql_connect
from kline_statistics import SHORT_TERM_SURGE_TYPE
from stock_selector import parse_date
from surge_pattern_miner import (
    DEFAULT_DAILY_WINDOW,
    DEFAULT_WEEKLY_WINDOW,
    PATTERN_COLUMNS,
    SurgePatternConfig,
    ensure_surge_pattern_table,
    extract_features_at_position,
    extract_features_for_date,
    iter_batches,
    load_kline_for_symbols,
    load_symbols,
    make_frame_map,
    pattern_to_text,
    save_patterns,
)


@dataclass(frozen=True)
class GoalSearchConfig:
    train_start_date: date
    train_end_date: date
    holdout_start_date: date
    holdout_end_date: date
    stat_type: str
    target_success_rate: float
    seed: int
    negative_ratio: float
    min_eval_success_rate: float
    min_eval_sample_count: int
    min_holdout_sample_count: int
    min_positive_supports: tuple[int, ...]
    max_pattern_sizes: tuple[int, ...]
    max_candidates: int
    daily_window: int
    weekly_window: int
    batch_size: int
    output: Path | None
    save_db: bool


def stable_rank(seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in (seed, *parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def split_positive_events(events: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = events.copy()
    ordered["_SplitRank"] = ordered.apply(lambda row: stable_rank(seed, row["SCode"], row["PrevTradeDate"]), axis=1)
    ordered = ordered.sort_values("_SplitRank").reset_index(drop=True)
    eval_count = max(1, int(round(len(ordered) * 0.2))) if len(ordered) > 1 else 0
    eval_events = ordered.iloc[:eval_count].drop(columns=["_SplitRank"]).reset_index(drop=True)
    train_events = ordered.iloc[eval_count:].drop(columns=["_SplitRank"]).reset_index(drop=True)
    return train_events, eval_events


def load_positive_events(conn, stat_type: str, start_date: date, end_date: date) -> pd.DataFrame:
    sql = """
        SELECT SCode, SName, StartRiseDate, PrevTradeDate, GainRate
        FROM klinestatistics
        WHERE StatType = %s
          AND PrevTradeDate >= %s
          AND PrevTradeDate <= %s
        ORDER BY SCode, PrevTradeDate
    """
    with conn.cursor() as cur:
        cur.execute(sql, (stat_type, start_date, end_date))
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["SCode", "SName", "StartRiseDate", "PrevTradeDate", "GainRate"])
    if df.empty:
        return df
    for column in ["StartRiseDate", "PrevTradeDate"]:
        df[column] = pd.to_datetime(df[column]).dt.date
    df["GainRate"] = pd.to_numeric(df["GainRate"], errors="coerce")
    return df


def load_negative_events(conn, stat_type: str, start_date: date, end_date: date, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0:
        return pd.DataFrame(columns=["SCode", "SName", "PrevTradeDate", "GainRate"])
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT SCode, PrevTradeDate
            FROM klinestatistics
            WHERE StatType = %s
              AND PrevTradeDate >= %s
              AND PrevTradeDate <= %s
            """,
            (stat_type, start_date, end_date),
        )
        positive_keys = {(str(row[0]), parse_date(row[1])) for row in cur.fetchall()}
    scan_limit = max(5000, limit * 20)
    sql = """
        SELECT dk.SCode, si.SName, DATE(dk.KTime) AS PrevTradeDate, 0.0 AS GainRate
        FROM dkandles dk
        LEFT JOIN stockinfo si ON si.SCode = dk.SCode
        WHERE dk.KType = %s
          AND dk.KTime >= %s
          AND dk.KTime < %s
        ORDER BY dk.SCode, dk.KTime
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (DEFAULT_KTYPE, start_date, end_date + timedelta(days=1), scan_limit))
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["SCode", "SName", "PrevTradeDate", "GainRate"])
    if df.empty:
        return df
    df["PrevTradeDate"] = pd.to_datetime(df["PrevTradeDate"]).dt.date
    df = df[~df.apply(lambda row: (str(row["SCode"]), row["PrevTradeDate"]) in positive_keys, axis=1)].copy()
    df["_Rank"] = df.apply(lambda row: stable_rank(seed, row["SCode"], row["PrevTradeDate"], "negative"), axis=1)
    return df.sort_values("_Rank").drop(columns=["_Rank"]).head(limit).reset_index(drop=True)


def event_key_frame(events: pd.DataFrame, label: int) -> pd.DataFrame:
    result = events[["SCode", "SName", "PrevTradeDate", "GainRate"]].copy()
    result["Label"] = label
    return result


def iter_pattern_keys_limited(features: set[str], max_pattern_size: int) -> Iterable[tuple[str, ...]]:
    ordered = sorted(features)
    for size in range(1, max_pattern_size + 1):
        yield from combinations(ordered, size)


def mine_support(conn, positives: pd.DataFrame, config: GoalSearchConfig, max_pattern_size: int) -> dict[tuple[str, ...], int]:
    support: dict[tuple[str, ...], int] = defaultdict(int)
    if positives.empty:
        return support
    symbols = sorted(positives["SCode"].dropna().unique().tolist())
    start_date = min(positives["PrevTradeDate"])
    end_date = max(positives["PrevTradeDate"])
    lookback_start = start_date - timedelta(days=max(500, config.weekly_window * 10, config.daily_window * 3))
    for batch_index, batch in enumerate(iter_batches(symbols, config.batch_size), start=1):
        daily_df = load_kline_for_symbols(conn, "dkandles", DEFAULT_KTYPE, batch, lookback_start, end_date)
        weekly_df = load_kline_for_symbols(conn, "wkandles", "W", batch, lookback_start, end_date)
        daily_frames = make_frame_map(daily_df)
        weekly_frames = make_frame_map(weekly_df)
        batch_events = positives[positives["SCode"].isin(batch)]
        feature_rows = 0
        for event in batch_events.itertuples(index=False):
            daily_frame = daily_frames.get(event.SCode)
            weekly_frame = weekly_frames.get(event.SCode)
            if daily_frame is None or weekly_frame is None:
                continue
            features = extract_features_for_date(daily_frame, weekly_frame, event.PrevTradeDate, config.daily_window, config.weekly_window)
            if not features:
                continue
            feature_rows += 1
            for pattern in iter_pattern_keys_limited(features, max_pattern_size):
                support[pattern] += 1
        print(f"mine batch {batch_index} symbols={len(batch)} events={len(batch_events)} feature_rows={feature_rows} patterns={len(support)}", flush=True)
    return support


def evaluate_event_sample(
    conn,
    events: pd.DataFrame,
    target_patterns: set[tuple[str, ...]],
    config: GoalSearchConfig,
    max_pattern_size: int,
) -> pd.DataFrame:
    counts = {pattern: [0, 0] for pattern in target_patterns}
    if events.empty or not target_patterns:
        return pattern_rows(counts, {}, config.target_success_rate, config.min_eval_sample_count)
    symbols = sorted(events["SCode"].dropna().unique().tolist())
    start_date = min(events["PrevTradeDate"])
    end_date = max(events["PrevTradeDate"])
    lookback_start = start_date - timedelta(days=max(500, config.weekly_window * 10, config.daily_window * 3))
    target_by_size: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    for pattern in target_patterns:
        target_by_size[len(pattern)].add(pattern)
    for batch_index, batch in enumerate(iter_batches(symbols, config.batch_size), start=1):
        daily_df = load_kline_for_symbols(conn, "dkandles", DEFAULT_KTYPE, batch, lookback_start, end_date)
        weekly_df = load_kline_for_symbols(conn, "wkandles", "W", batch, lookback_start, end_date)
        daily_frames = make_frame_map(daily_df)
        weekly_frames = make_frame_map(weekly_df)
        batch_events = events[events["SCode"].isin(batch)]
        hits = 0
        for event in batch_events.itertuples(index=False):
            daily_frame = daily_frames.get(event.SCode)
            weekly_frame = weekly_frames.get(event.SCode)
            if daily_frame is None or weekly_frame is None:
                continue
            features = extract_features_for_date(daily_frame, weekly_frame, event.PrevTradeDate, config.daily_window, config.weekly_window)
            if not features:
                continue
            seen = matching_patterns(features, target_by_size, max_pattern_size)
            for pattern in seen:
                counts[pattern][0] += 1
                counts[pattern][1] += int(event.Label == 1)
                hits += 1
        print(f"eval-sample batch {batch_index} symbols={len(batch)} events={len(batch_events)} hits={hits}", flush=True)
    return pattern_rows(counts, {}, config.min_eval_success_rate, config.min_eval_sample_count)


def matching_patterns(features: set[str], target_by_size: dict[int, set[tuple[str, ...]]], max_pattern_size: int) -> set[tuple[str, ...]]:
    ordered = sorted(features)
    matched: set[tuple[str, ...]] = set()
    for size in range(1, max_pattern_size + 1):
        targets = target_by_size.get(size)
        if not targets:
            continue
        for pattern in combinations(ordered, size):
            if pattern in targets:
                matched.add(pattern)
    return matched


def evaluate_holdout_universe(
    conn,
    positives: pd.DataFrame,
    target_patterns: set[tuple[str, ...]],
    positive_support: dict[tuple[str, ...], int],
    config: GoalSearchConfig,
    max_pattern_size: int,
) -> pd.DataFrame:
    counts = {pattern: [0, 0] for pattern in target_patterns}
    positive_keys = set(zip(positives["SCode"], positives["PrevTradeDate"]))
    target_by_size: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    for pattern in target_patterns:
        target_by_size[len(pattern)].add(pattern)
    lookback_start = config.holdout_start_date - timedelta(days=max(500, config.weekly_window * 10, config.daily_window * 3))
    symbols = load_symbols(conn, config.holdout_start_date, config.holdout_end_date, DEFAULT_KTYPE)
    batches = list(iter_batches(symbols, config.batch_size))
    for batch_index, batch in enumerate(batches, start=1):
        daily_df = load_kline_for_symbols(conn, "dkandles", DEFAULT_KTYPE, batch, lookback_start, config.holdout_end_date)
        weekly_df = load_kline_for_symbols(conn, "wkandles", "W", batch, lookback_start, config.holdout_end_date)
        daily_frames = make_frame_map(daily_df)
        weekly_frames = make_frame_map(weekly_df)
        scanned = 0
        hits = 0
        for symbol in batch:
            daily_frame = daily_frames.get(symbol)
            weekly_frame = weekly_frames.get(symbol)
            if daily_frame is None or weekly_frame is None:
                continue
            date_rows = daily_frame[(daily_frame["TradeDate"] >= config.holdout_start_date) & (daily_frame["TradeDate"] <= config.holdout_end_date)]
            for daily_pos, item in date_rows.iterrows():
                features = extract_features_at_position(
                    daily_frame,
                    weekly_frame,
                    int(daily_pos),
                    item.TradeDate,
                    config.daily_window,
                    config.weekly_window,
                )
                if not features:
                    continue
                scanned += 1
                is_success = (symbol, item.TradeDate) in positive_keys
                for pattern in matching_patterns(features, target_by_size, max_pattern_size):
                    counts[pattern][0] += 1
                    counts[pattern][1] += int(is_success)
                    hits += 1
        print(f"holdout batch {batch_index}/{len(batches)} symbols={len(batch)} scanned={scanned} hits={hits}", flush=True)
    return pattern_rows(counts, positive_support, config.target_success_rate, config.min_holdout_sample_count)


def pattern_rows(
    counts: dict[tuple[str, ...], list[int]],
    positive_support: dict[tuple[str, ...], int],
    min_success_rate: float,
    min_sample_count: int,
) -> pd.DataFrame:
    rows = []
    for pattern, (sample_count, success_count) in counts.items():
        if sample_count < min_sample_count:
            continue
        success_rate = success_count / sample_count if sample_count else 0.0
        if success_rate < min_success_rate:
            continue
        rows.append(
            {
                "MinSuccessRate": min_success_rate,
                "Pattern": pattern_to_text(pattern),
                "FeatureCount": len(pattern),
                "SampleCount": sample_count,
                "SuccessCount": success_count,
                "SuccessRate": success_rate,
                "PositiveSupport": positive_support.get(pattern, 0),
            }
        )
    if not rows:
        return pd.DataFrame(columns=PATTERN_COLUMNS)
    return pd.DataFrame(rows, columns=PATTERN_COLUMNS).sort_values(
        ["SuccessRate", "SuccessCount", "SampleCount", "PositiveSupport"],
        ascending=[False, False, False, False],
    )


def run_goal_search(config: GoalSearchConfig) -> pd.DataFrame:
    with mysql_connect() as conn:
        if config.save_db:
            ensure_surge_pattern_table(conn)
        positives = load_positive_events(conn, config.stat_type, config.train_start_date, config.train_end_date)
        holdout_positives = load_positive_events(conn, config.stat_type, config.holdout_start_date, config.holdout_end_date)
        train_pos, eval_pos = split_positive_events(positives, config.seed)
        print(
            f"loaded positive rows train_total={len(positives)} train_80={len(train_pos)} eval_20={len(eval_pos)} "
            f"holdout_pos={len(holdout_positives)}; loading eval negatives...",
            flush=True,
        )
        eval_neg = load_negative_events(
            conn,
            config.stat_type,
            config.train_start_date,
            config.train_end_date,
            max(1, int(len(eval_pos) * config.negative_ratio)),
            config.seed,
        )
        eval_events = pd.concat([event_key_frame(eval_pos, 1), event_key_frame(eval_neg, 0)], ignore_index=True)
        eval_events["_Rank"] = eval_events.apply(lambda row: stable_rank(config.seed, row["SCode"], row["PrevTradeDate"], row["Label"]), axis=1)
        eval_events = eval_events.sort_values("_Rank").drop(columns=["_Rank"]).reset_index(drop=True)
        print(
            f"loaded positives train_total={len(positives)} train_80={len(train_pos)} eval_20={len(eval_pos)} "
            f"eval_neg={len(eval_neg)} holdout_pos={len(holdout_positives)}",
            flush=True,
        )
        best = pd.DataFrame(columns=PATTERN_COLUMNS)
        evaluated_target_sets: set[frozenset[tuple[str, ...]]] = set()
        for max_pattern_size in config.max_pattern_sizes:
            print(f"search max_pattern_size={max_pattern_size}", flush=True)
            support = mine_support(conn, train_pos, config, max_pattern_size)
            for min_support in config.min_positive_supports:
                target_patterns = {pattern for pattern, count in support.items() if count >= min_support}
                print(f"candidate min_support={min_support} target_patterns={len(target_patterns)}", flush=True)
                if not target_patterns:
                    continue
                fingerprint = frozenset(target_patterns)
                if fingerprint in evaluated_target_sets:
                    print("candidate set already evaluated; skipping duplicate", flush=True)
                    continue
                evaluated_target_sets.add(fingerprint)
                eval_patterns = evaluate_event_sample(conn, eval_events, target_patterns, config, max_pattern_size)
                if eval_patterns.empty:
                    print("internal eval kept=0", flush=True)
                    continue
                eval_patterns = eval_patterns.head(config.max_candidates)
                selected = {tuple(str(row.Pattern).split(" && ")) for row in eval_patterns.itertuples(index=False)}
                print(f"internal eval kept={len(eval_patterns)}; validating holdout candidates={len(selected)}", flush=True)
                holdout = evaluate_holdout_universe(conn, holdout_positives, selected, support, config, max_pattern_size)
                if not holdout.empty:
                    best = holdout
                    if config.output:
                        config.output.parent.mkdir(parents=True, exist_ok=True)
                        best.to_csv(config.output, index=False, encoding="utf-8-sig")
                    saved = 0
                    if config.save_db:
                        saved = save_patterns(
                            conn,
                            best,
                            SurgePatternConfig(
                                test_start_date=config.holdout_start_date.strftime("%Y%m%d"),
                                test_end_date=config.holdout_end_date.strftime("%Y%m%d"),
                                train_start_date=config.train_start_date.strftime("%Y%m%d"),
                                train_end_date=config.train_end_date.strftime("%Y%m%d"),
                                stat_type=config.stat_type,
                                min_success_rates=(config.target_success_rate,),
                                min_sample_count=config.min_holdout_sample_count,
                                min_positive_support=min_support,
                                max_pattern_size=max_pattern_size,
                                daily_window=config.daily_window,
                                weekly_window=config.weekly_window,
                                batch_size=config.batch_size,
                                output=str(config.output) if config.output else None,
                                save_db=config.save_db,
                            ),
                            config.train_start_date,
                            config.train_end_date,
                            config.holdout_start_date,
                            config.holdout_end_date,
                        )
                    print(f"GOAL REACHED patterns={len(best)} saved={saved}", flush=True)
                    print(best.head(20).to_string(index=False), flush=True)
                    return best
        print("goal not reached", flush=True)
        return best


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatically search a validated surge pattern with a holdout success-rate target")
    parser.add_argument("--train-start-date", default="20200101")
    parser.add_argument("--train-end-date", default="20251231")
    parser.add_argument("--holdout-start-date", default="20260101")
    parser.add_argument("--holdout-end-date", default="20260430")
    parser.add_argument("--stat-type", default=SHORT_TERM_SURGE_TYPE)
    parser.add_argument("--target-success-rate", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--min-eval-success-rate", type=float, default=0.25)
    parser.add_argument("--min-eval-sample-count", type=int, default=20)
    parser.add_argument("--min-holdout-sample-count", type=int, default=5)
    parser.add_argument("--min-positive-supports", type=parse_int_list, default=parse_int_list("1000,500,200,100,50,20,10,5"))
    parser.add_argument("--max-pattern-sizes", type=parse_int_list, default=parse_int_list("1,2,3"))
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--daily-window", type=int, default=DEFAULT_DAILY_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WEEKLY_WINDOW)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("data") / "goal_patterns_2020_2025_to_2026q1.csv")
    parser.add_argument("--no-save-db", action="store_true")
    return parser


def normalize_rate(value: float) -> float:
    return value / 100 if value > 1 else value


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_goal_search(
        GoalSearchConfig(
            train_start_date=parse_date(args.train_start_date),
            train_end_date=parse_date(args.train_end_date),
            holdout_start_date=parse_date(args.holdout_start_date),
            holdout_end_date=parse_date(args.holdout_end_date),
            stat_type=args.stat_type,
            target_success_rate=normalize_rate(args.target_success_rate),
            seed=args.seed,
            negative_ratio=max(0.0, args.negative_ratio),
            min_eval_success_rate=normalize_rate(args.min_eval_success_rate),
            min_eval_sample_count=max(1, args.min_eval_sample_count),
            min_holdout_sample_count=max(1, args.min_holdout_sample_count),
            min_positive_supports=args.min_positive_supports,
            max_pattern_sizes=args.max_pattern_sizes,
            max_candidates=max(1, args.max_candidates),
            daily_window=max(2, args.daily_window),
            weekly_window=max(2, args.weekly_window),
            batch_size=max(1, args.batch_size),
            output=args.output,
            save_db=not args.no_save_db,
        )
    )


if __name__ == "__main__":
    main()
