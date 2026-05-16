from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from a_share_crawler import DEFAULT_KTYPE, mysql_connect
from goal_pattern_search import (
    event_key_frame,
    load_negative_events,
    load_positive_events,
    split_positive_events,
    stable_rank,
)
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
class SignatureSearchConfig:
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
    max_candidates: int
    daily_window: int
    weekly_window: int
    batch_size: int
    output: Path | None
    save_db: bool


def feature_signature(features: set[str]) -> tuple[str, ...]:
    return tuple(sorted(features))


def mine_signatures(conn, positives: pd.DataFrame, config: SignatureSearchConfig) -> dict[tuple[str, ...], int]:
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
        rows = 0
        for event in batch_events.itertuples(index=False):
            daily_frame = daily_frames.get(event.SCode)
            weekly_frame = weekly_frames.get(event.SCode)
            if daily_frame is None or weekly_frame is None:
                continue
            features = extract_features_for_date(daily_frame, weekly_frame, event.PrevTradeDate, config.daily_window, config.weekly_window)
            if not features:
                continue
            support[feature_signature(features)] += 1
            rows += 1
        print(f"signature mine batch {batch_index} symbols={len(batch)} events={len(batch_events)} rows={rows} signatures={len(support)}", flush=True)
    return support


def rows_from_counts(
    counts: dict[tuple[str, ...], list[int]],
    support: dict[tuple[str, ...], int],
    min_rate: float,
    min_sample_count: int,
) -> pd.DataFrame:
    rows = []
    for pattern, (sample_count, success_count) in counts.items():
        if sample_count < min_sample_count:
            continue
        rate = success_count / sample_count if sample_count else 0.0
        if rate < min_rate:
            continue
        rows.append(
            {
                "MinSuccessRate": min_rate,
                "Pattern": pattern_to_text(pattern),
                "FeatureCount": len(pattern),
                "SampleCount": sample_count,
                "SuccessCount": success_count,
                "SuccessRate": rate,
                "PositiveSupport": support.get(pattern, 0),
            }
        )
    if not rows:
        return pd.DataFrame(columns=PATTERN_COLUMNS)
    return pd.DataFrame(rows, columns=PATTERN_COLUMNS).sort_values(
        ["SuccessRate", "SuccessCount", "SampleCount", "PositiveSupport"],
        ascending=[False, False, False, False],
    )


def evaluate_sample(conn, events: pd.DataFrame, candidates: set[tuple[str, ...]], support: dict[tuple[str, ...], int], config: SignatureSearchConfig) -> pd.DataFrame:
    counts = {pattern: [0, 0] for pattern in candidates}
    if events.empty or not candidates:
        return rows_from_counts(counts, support, config.min_eval_success_rate, config.min_eval_sample_count)
    symbols = sorted(events["SCode"].dropna().unique().tolist())
    start_date = min(events["PrevTradeDate"])
    end_date = max(events["PrevTradeDate"])
    lookback_start = start_date - timedelta(days=max(500, config.weekly_window * 10, config.daily_window * 3))
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
            signature = feature_signature(features)
            if signature not in candidates:
                continue
            counts[signature][0] += 1
            counts[signature][1] += int(event.Label == 1)
            hits += 1
        print(f"signature eval batch {batch_index} symbols={len(batch)} events={len(batch_events)} hits={hits}", flush=True)
    return rows_from_counts(counts, support, config.min_eval_success_rate, config.min_eval_sample_count)


def evaluate_holdout(conn, positives: pd.DataFrame, candidates: set[tuple[str, ...]], support: dict[tuple[str, ...], int], config: SignatureSearchConfig) -> pd.DataFrame:
    counts = {pattern: [0, 0] for pattern in candidates}
    positive_keys = set(zip(positives["SCode"], positives["PrevTradeDate"]))
    lookback_start = config.holdout_start_date - timedelta(days=max(500, config.weekly_window * 10, config.daily_window * 3))
    symbols = load_symbols(conn, config.holdout_start_date, config.holdout_end_date, DEFAULT_KTYPE)
    batches = list(iter_batches(symbols, config.batch_size))
    for batch_index, batch in enumerate(batches, start=1):
        daily_df = load_kline_for_symbols(conn, "dkandles", DEFAULT_KTYPE, batch, lookback_start, config.holdout_end_date)
        weekly_df = load_kline_for_symbols(conn, "wkandles", "W", batch, lookback_start, config.holdout_end_date)
        daily_frames = make_frame_map(daily_df)
        weekly_frames = make_frame_map(weekly_df)
        scanned = hits = 0
        for symbol in batch:
            daily_frame = daily_frames.get(symbol)
            weekly_frame = weekly_frames.get(symbol)
            if daily_frame is None or weekly_frame is None:
                continue
            date_rows = daily_frame[(daily_frame["TradeDate"] >= config.holdout_start_date) & (daily_frame["TradeDate"] <= config.holdout_end_date)]
            for daily_pos, item in date_rows.iterrows():
                features = extract_features_at_position(daily_frame, weekly_frame, int(daily_pos), item.TradeDate, config.daily_window, config.weekly_window)
                if not features:
                    continue
                scanned += 1
                signature = feature_signature(features)
                if signature not in candidates:
                    continue
                counts[signature][0] += 1
                counts[signature][1] += int((symbol, item.TradeDate) in positive_keys)
                hits += 1
        print(f"signature holdout batch {batch_index}/{len(batches)} symbols={len(batch)} scanned={scanned} hits={hits}", flush=True)
    return rows_from_counts(counts, support, config.target_success_rate, config.min_holdout_sample_count)


def run_signature_search(config: SignatureSearchConfig) -> pd.DataFrame:
    with mysql_connect() as conn:
        if config.save_db:
            ensure_surge_pattern_table(conn)
        positives = load_positive_events(conn, config.stat_type, config.train_start_date, config.train_end_date)
        holdout_positives = load_positive_events(conn, config.stat_type, config.holdout_start_date, config.holdout_end_date)
        train_pos, eval_pos = split_positive_events(positives, config.seed)
        print(
            f"loaded positives train_total={len(positives)} train_80={len(train_pos)} eval_20={len(eval_pos)} "
            f"holdout_pos={len(holdout_positives)}",
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
        support = mine_signatures(conn, train_pos, config)
        seen: set[frozenset[tuple[str, ...]]] = set()
        for min_support in config.min_positive_supports:
            candidates = {pattern for pattern, count in support.items() if count >= min_support}
            print(f"signature candidates min_support={min_support} count={len(candidates)}", flush=True)
            if not candidates:
                continue
            fingerprint = frozenset(candidates)
            if fingerprint in seen:
                print("signature candidate set already evaluated; skipping", flush=True)
                continue
            seen.add(fingerprint)
            eval_patterns = evaluate_sample(conn, eval_events, candidates, support, config)
            if eval_patterns.empty:
                print("signature internal eval kept=0", flush=True)
                continue
            eval_patterns = eval_patterns.head(config.max_candidates)
            selected = {tuple(str(row.Pattern).split(" && ")) for row in eval_patterns.itertuples(index=False)}
            print(f"signature internal eval kept={len(eval_patterns)}; validating holdout candidates={len(selected)}", flush=True)
            holdout = evaluate_holdout(conn, holdout_positives, selected, support, config)
            if holdout.empty:
                continue
            if config.output:
                config.output.parent.mkdir(parents=True, exist_ok=True)
                holdout.to_csv(config.output, index=False, encoding="utf-8-sig")
            saved = 0
            if config.save_db:
                saved = save_patterns(
                    conn,
                    holdout,
                    SurgePatternConfig(
                        test_start_date=config.holdout_start_date.strftime("%Y%m%d"),
                        test_end_date=config.holdout_end_date.strftime("%Y%m%d"),
                        train_start_date=config.train_start_date.strftime("%Y%m%d"),
                        train_end_date=config.train_end_date.strftime("%Y%m%d"),
                        stat_type=config.stat_type,
                        min_success_rates=(config.target_success_rate,),
                        min_sample_count=config.min_holdout_sample_count,
                        min_positive_support=min_support,
                        max_pattern_size=99,
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
            print(f"SIGNATURE GOAL REACHED patterns={len(holdout)} saved={saved}", flush=True)
            print(holdout.head(20).to_string(index=False), flush=True)
            return holdout
        print("signature goal not reached", flush=True)
        return pd.DataFrame(columns=PATTERN_COLUMNS)


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def normalize_rate(value: float) -> float:
    return value / 100 if value > 1 else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search full-feature signatures that meet a holdout success-rate target")
    parser.add_argument("--train-start-date", default="20200101")
    parser.add_argument("--train-end-date", default="20251231")
    parser.add_argument("--holdout-start-date", default="20260101")
    parser.add_argument("--holdout-end-date", default="20260430")
    parser.add_argument("--stat-type", default=SHORT_TERM_SURGE_TYPE)
    parser.add_argument("--target-success-rate", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--min-eval-success-rate", type=float, default=0.25)
    parser.add_argument("--min-eval-sample-count", type=int, default=5)
    parser.add_argument("--min-holdout-sample-count", type=int, default=5)
    parser.add_argument("--min-positive-supports", type=parse_int_list, default=parse_int_list("200,100,50,20,10,5,3,2,1"))
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--daily-window", type=int, default=DEFAULT_DAILY_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WEEKLY_WINDOW)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("data") / "signature_goal_patterns_2020_2025_to_2026q1.csv")
    parser.add_argument("--no-save-db", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_signature_search(
        SignatureSearchConfig(
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
