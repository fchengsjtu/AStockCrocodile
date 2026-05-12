from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd

from a_share_crawler import DEFAULT_KTYPE, load_env_file, mysql_connect
from kline_statistics import SHORT_TERM_SURGE_TYPE, ensure_kline_statistics_table
from llm_surge_pattern_miner import (
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY_ENV,
    DEFAULT_LOCAL_API_KEY,
    DEFAULT_MODEL,
    call_llm_chat,
    fallback_patterns_from_counts,
    parse_llm_patterns,
)
from surge_pattern_miner import (
    DEFAULT_BATCH_SIZE,
    SURGE_PATTERN_TABLE,
    SurgePatternConfig,
    backfill_kline_stat_selection_dates,
    evaluate_patterns,
    extract_features_for_date,
    ensure_surge_pattern_table,
    iter_batches,
    load_kline_for_symbols,
    make_frame_map,
    pattern_to_text,
    save_patterns,
)

DEFAULT_WINDOW = 55
DEFAULT_SPLIT_SEED = 20260512
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_MIN_SUCCESS_RATE = 0.20
DEFAULT_PROMPT_BATCH_SIZE = 3
DEFAULT_MAX_TRAINING_BATCHES = 0
DEFAULT_BLACKBOX_CANDIDATE_COUNT = 12


@dataclass(frozen=True)
class BlackboxTrainingConfig:
    stat_type: str
    train_ratio: float
    split_seed: int
    daily_window: int
    weekly_window: int
    min_success_rate: float
    min_sample_count: int
    min_positive_support: int
    min_pattern_size: int
    max_pattern_size: int
    candidate_count: int
    batch_size: int
    prompt_batch_size: int
    max_training_batches: int
    model: str
    api_base_url: str
    api_key_env: str
    output: str | None
    save_db: bool


def _stable_sample_key(row) -> str:
    return f"{row.SCode}|{row.StartRiseDate}|{row.PrevTradeDate}"


def split_samples(samples: pd.DataFrame, train_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if samples.empty:
        return samples.copy(), samples.copy()
    ratio = min(max(train_ratio, 0.01), 0.99)
    ordered = samples.copy()
    ordered["_SplitRank"] = ordered.apply(
        lambda row: int(hashlib.sha256(f"{seed}|{_stable_sample_key(row)}".encode("utf-8")).hexdigest()[:16], 16),
        axis=1,
    )
    ordered = ordered.sort_values(["_SplitRank", "SCode", "PrevTradeDate", "StartRiseDate"]).reset_index(drop=True)
    split_at = int(round(len(ordered) * ratio))
    split_at = min(max(split_at, 1), len(ordered) - 1) if len(ordered) > 1 else len(ordered)
    train = ordered.iloc[:split_at].drop(columns=["_SplitRank"]).reset_index(drop=True)
    test = ordered.iloc[split_at:].drop(columns=["_SplitRank"]).reset_index(drop=True)
    return train, test


def load_kline_stat_samples(conn, stat_type: str) -> pd.DataFrame:
    sql = """
        SELECT SCode, SName, StartRiseDate, PrevTradeDate, COALESCE(SelectionDate, PrevTradeDate) AS SelectionDate, GainRate
        FROM klinestatistics
        WHERE StatType = %s
          AND PrevTradeDate IS NOT NULL
        ORDER BY SCode, PrevTradeDate, StartRiseDate
    """
    with conn.cursor() as cur:
        cur.execute(sql, (stat_type,))
        rows = cur.fetchall()
    columns = ["SCode", "SName", "StartRiseDate", "PrevTradeDate", "SelectionDate", "GainRate"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    for column in ["StartRiseDate", "PrevTradeDate", "SelectionDate"]:
        df[column] = pd.to_datetime(df[column]).dt.date
    return df


def _round_float(value) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 6)


def _compact_window_to_matrix(frame: pd.DataFrame) -> dict:
    ordered = frame.sort_values("TradeDate").reset_index(drop=True)
    if ordered.empty:
        return {"start": None, "end": None, "columns": ["o", "h", "l", "c", "v"], "scale": {}, "rows": []}
    anchor_close = float(ordered.iloc[-1]["Close"])
    avg_volume = float(ordered["Volume"].mean(skipna=True) or 0)
    rows = []
    for item in ordered.itertuples(index=False):
        open_value = float(item.Open)
        high_value = float(item.High)
        low_value = float(item.Low)
        close_value = float(item.Close)
        volume = float(item.Volume)
        rows.append(
            [
                round((open_value / anchor_close - 1) * 10000) if anchor_close > 0 else None,
                round((high_value / anchor_close - 1) * 10000) if anchor_close > 0 else None,
                round((low_value / anchor_close - 1) * 10000) if anchor_close > 0 else None,
                round((close_value / anchor_close - 1) * 10000) if anchor_close > 0 else None,
                round(volume / avg_volume * 100) if avg_volume > 0 else None,
            ]
        )
    return {
        "start": str(ordered.iloc[0]["TradeDate"]),
        "end": str(ordered.iloc[-1]["TradeDate"]),
        "columns": ["open_bp", "high_bp", "low_bp", "close_bp", "volume_pct_avg"],
        "scale": {
            "price_bp": "basis points versus last close in this window",
            "volume_pct_avg": "volume divided by this window's average volume, percent",
        },
        "rows": rows,
    }


def collect_raw_samples_for_events(conn, events: pd.DataFrame, config: BlackboxTrainingConfig) -> tuple[list[dict], Counter]:
    samples: list[dict] = []
    feature_counts: Counter[tuple[str, ...]] = Counter()
    if events.empty:
        return samples, feature_counts

    start_date = min(events["PrevTradeDate"])
    end_date = max(events["PrevTradeDate"])
    lookback_start = start_date - timedelta(days=max(500, config.weekly_window * 10, config.daily_window * 3))
    symbols = sorted(events["SCode"].dropna().unique().tolist())
    for batch in iter_batches(symbols, config.batch_size):
        daily_df = load_kline_for_symbols(conn, "dkandles", DEFAULT_KTYPE, batch, lookback_start, end_date)
        weekly_df = load_kline_for_symbols(conn, "wkandles", "W", batch, lookback_start, end_date)
        daily_frames = make_frame_map(daily_df)
        weekly_frames = make_frame_map(weekly_df)
        batch_events = events[events["SCode"].isin(batch)]
        for event in batch_events.itertuples(index=False):
            daily_frame = daily_frames.get(event.SCode)
            weekly_frame = weekly_frames.get(event.SCode)
            if daily_frame is None or weekly_frame is None:
                continue
            daily_matches = daily_frame.index[daily_frame["TradeDate"] == event.PrevTradeDate].tolist()
            if not daily_matches:
                continue
            daily_pos = daily_matches[0]
            weekly_positions = weekly_frame.index[weekly_frame["TradeDate"] <= event.PrevTradeDate].tolist()
            if not weekly_positions:
                continue
            weekly_pos = weekly_positions[-1]
            if daily_pos + 1 < config.daily_window or weekly_pos + 1 < config.weekly_window:
                continue

            daily_window = daily_frame.iloc[daily_pos + 1 - config.daily_window : daily_pos + 1]
            weekly_window = weekly_frame.iloc[weekly_pos + 1 - config.weekly_window : weekly_pos + 1]
            features = extract_features_for_date(
                daily_frame,
                weekly_frame,
                event.PrevTradeDate,
                config.daily_window,
                config.weekly_window,
            )
            if not features:
                continue
            for feature in sorted(features):
                feature_counts[(feature,)] += 1

            samples.append(
                {
                    "scode": event.SCode,
                    "anchor_date": str(event.PrevTradeDate),
                    "start_rise_date": str(event.StartRiseDate),
                    "gain_rate": _round_float(event.GainRate),
                    "daily_55": _compact_window_to_matrix(daily_window),
                    "weekly_55": _compact_window_to_matrix(weekly_window),
                }
            )
    return samples, feature_counts


def build_blackbox_prompt(samples: list[dict], feature_counts: Counter, config: BlackboxTrainingConfig, batch_no: int) -> str:
    allowed_features = sorted(feature[0] for feature in feature_counts if len(feature) == 1)
    payload = {
        "task": "Black-box technical pattern training from positive klinestatistics samples.",
        "stat_type": config.stat_type,
        "batch_no": batch_no,
        "label": "Every sample is a positive short-term surge setup.",
        "input_anchor": "Use PrevTradeDate as anchor_date. daily_55 ends on PrevTradeDate. weekly_55 ends on the latest weekly K-line on or before PrevTradeDate.",
        "compact_kline_format": "daily_55 and weekly_55 each contain 55 OHLCV rows. Price values are basis points versus the last close in that window. Volume is percent of the window average volume.",
        "future_use": "A saved pattern will be applied to any stock at any date using its previous 55 daily bars and previous 55 weekly bars.",
        "minimum_required_validated_success_rate": config.min_success_rate,
        "min_features_per_pattern": config.min_pattern_size,
        "max_features_per_pattern": config.max_pattern_size,
        "candidate_count": config.candidate_count,
        "allowed_feature_tokens": allowed_features,
        "positive_samples": samples,
    }
    return (
        "You are a quantitative trading research assistant. "
        "Infer reusable technical setup patterns from the raw K-line windows. "
        "Return only strict JSON. Every feature must be copied exactly from allowed_feature_tokens, "
        "because the program validates patterns mechanically on out-of-sample data. "
        "Use 3 to 8 features per pattern unless configured otherwise. "
        "Schema: "
        '{"patterns":[{"name":"short_name","features":["TOKEN_A","TOKEN_B","TOKEN_C"],"rationale":"brief reason"}]}. '
        "No markdown, no commentary.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def generate_blackbox_patterns(conn, train_events: pd.DataFrame, config: BlackboxTrainingConfig) -> tuple[set[tuple[str, ...]], Counter]:
    all_patterns: set[tuple[str, ...]] = set()
    positive_support: Counter[tuple[str, ...]] = Counter()
    event_batches = list(iter_batches(list(train_events.index), config.prompt_batch_size))
    if config.max_training_batches > 0:
        event_batches = event_batches[: config.max_training_batches]

    def train_batch(batch_events: pd.DataFrame, batch_label: str) -> list[tuple[str, ...]]:
        samples, feature_counts = collect_raw_samples_for_events(conn, batch_events, config)
        valid_features = {feature[0] for feature in feature_counts if len(feature) == 1}
        if not samples or not valid_features:
            print(f"blackbox train batch {batch_label} skipped samples={len(samples)}", flush=True)
            return []
        prompt = build_blackbox_prompt(samples, feature_counts, config, int(batch_label.split(".")[0]))
        llm_config = type(
            "LlmCallConfig",
            (),
            {
                "api_key_env": config.api_key_env,
                "api_base_url": config.api_base_url,
                "model": config.model,
            },
        )()
        try:
            response_text = call_llm_chat(prompt, llm_config)
            patterns = parse_llm_patterns(response_text, valid_features, config.min_pattern_size, config.max_pattern_size)
        except Exception as exc:
            if len(batch_events) > 1:
                midpoint = len(batch_events) // 2
                print(
                    f"blackbox train batch {batch_label} LLM request failed; splitting "
                    f"{len(batch_events)} samples: {exc}",
                    flush=True,
                )
                left = train_batch(batch_events.iloc[:midpoint].reset_index(drop=True), f"{batch_label}.1")
                right = train_batch(batch_events.iloc[midpoint:].reset_index(drop=True), f"{batch_label}.2")
                return left + right
            print(
                f"blackbox train batch {batch_label} single-sample LLM request failed; using fallback features: {exc}",
                flush=True,
            )
            patterns = []
        if not patterns:
            patterns = fallback_patterns_from_counts(
                feature_counts,
                config.min_pattern_size,
                config.max_pattern_size,
                config.candidate_count,
            )
        for pattern in patterns[: config.candidate_count]:
            support_values = [feature_counts.get((feature,), 0) for feature in pattern]
            if support_values:
                positive_support[pattern] += min(support_values)
        print(
            f"blackbox train batch {batch_label} samples={len(samples)} "
            f"patterns={len(patterns)}",
            flush=True,
        )
        return patterns[: config.candidate_count]

    for batch_no, indexes in enumerate(event_batches, start=1):
        batch_events = train_events.loc[indexes].reset_index(drop=True)
        patterns = train_batch(batch_events, str(batch_no))
        for pattern in patterns:
            all_patterns.add(pattern)
        print(
            f"blackbox train batch {batch_no}/{len(event_batches)} "
            f"patterns_total={len(all_patterns)}",
            flush=True,
        )
    return all_patterns, positive_support


def _events_as_anchor_labels(events: pd.DataFrame) -> pd.DataFrame:
    labels = events.copy()
    labels["SelectionDate"] = labels["PrevTradeDate"]
    return labels


def run_blackbox_training(config: BlackboxTrainingConfig) -> pd.DataFrame:
    load_env_file()
    os.environ.setdefault(config.api_key_env, DEFAULT_LOCAL_API_KEY)
    with mysql_connect() as conn:
        ensure_kline_statistics_table(conn)
        if config.save_db:
            ensure_surge_pattern_table(conn)
        backfilled = backfill_kline_stat_selection_dates(conn, config.stat_type)
        if backfilled:
            print(f"backfilled SelectionDate rows={backfilled}", flush=True)

        samples = load_kline_stat_samples(conn, config.stat_type)
        train_events, test_events = split_samples(samples, config.train_ratio, config.split_seed)
        print(
            f"loaded klinestatistics samples={len(samples)} train={len(train_events)} test={len(test_events)} "
            f"stat_type={config.stat_type}",
            flush=True,
        )
        if train_events.empty or test_events.empty:
            raise RuntimeError("not enough klinestatistics samples for 80/20 black-box training")

        patterns, positive_support = generate_blackbox_patterns(conn, train_events, config)
        if not patterns:
            raise RuntimeError("local LLM did not produce any valid black-box patterns")

        test_start_date = min(test_events["PrevTradeDate"])
        test_end_date = max(test_events["PrevTradeDate"])
        labels = _events_as_anchor_labels(samples[(samples["PrevTradeDate"] >= test_start_date) & (samples["PrevTradeDate"] <= test_end_date)])
        pattern_config = SurgePatternConfig(
            test_start_date=str(test_start_date),
            test_end_date=str(test_end_date),
            train_start_date=str(min(train_events["PrevTradeDate"])),
            train_end_date=str(max(train_events["PrevTradeDate"])),
            stat_type=config.stat_type,
            min_success_rates=(config.min_success_rate,),
            min_sample_count=config.min_sample_count,
            min_positive_support=config.min_positive_support,
            max_pattern_size=config.max_pattern_size,
            daily_window=config.daily_window,
            weekly_window=config.weekly_window,
            batch_size=config.batch_size,
            output=config.output,
            save_db=config.save_db,
        )
        results = evaluate_patterns(
            conn,
            labels,
            patterns,
            positive_support,
            pattern_config,
            test_start_date,
            test_end_date,
        )
        if not results.empty:
            results = results[results["SuccessRate"] >= config.min_success_rate].copy()
        if config.output:
            results.to_csv(config.output, index=False, encoding="utf-8-sig")
        saved = save_patterns(
            conn,
            results,
            pattern_config,
            min(train_events["PrevTradeDate"]),
            max(train_events["PrevTradeDate"]),
            test_start_date,
            test_end_date,
        ) if config.save_db else 0
    print(
        f"blackbox patterns candidates={len(patterns)} kept={len(results)} saved={saved} "
        f"min_success_rate={config.min_success_rate:.2%} table={SURGE_PATTERN_TABLE}",
        flush=True,
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    load_env_file()
    parser = argparse.ArgumentParser(description="Black-box train reusable surge setup patterns with a local LLM and validate them on held-out klinestatistics samples")
    parser.add_argument("--stat-type", default=SHORT_TERM_SURGE_TYPE)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--daily-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-success-rate", type=float, default=DEFAULT_MIN_SUCCESS_RATE)
    parser.add_argument("--min-sample-count", type=int, default=20)
    parser.add_argument("--min-positive-support", type=int, default=5)
    parser.add_argument("--min-pattern-size", type=int, default=3)
    parser.add_argument("--max-pattern-size", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_BLACKBOX_CANDIDATE_COUNT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--prompt-batch-size", type=int, default=DEFAULT_PROMPT_BATCH_SIZE)
    parser.add_argument("--max-training-batches", type=int, default=DEFAULT_MAX_TRAINING_BATCHES, help="0 means use all training samples")
    parser.add_argument("--model", default=os.environ.get("LOCAL_LLM_MODEL") or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-base-url", default=os.environ.get("LOCAL_LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_API_BASE_URL))
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--output")
    parser.add_argument("--no-save-db", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    min_success_rate = args.min_success_rate / 100 if args.min_success_rate > 1 else args.min_success_rate
    run_blackbox_training(
        BlackboxTrainingConfig(
            stat_type=args.stat_type,
            train_ratio=args.train_ratio,
            split_seed=args.split_seed,
            daily_window=max(2, args.daily_window),
            weekly_window=max(2, args.weekly_window),
            min_success_rate=min_success_rate,
            min_sample_count=max(1, args.min_sample_count),
            min_positive_support=max(1, args.min_positive_support),
            min_pattern_size=max(1, args.min_pattern_size),
            max_pattern_size=max(max(1, args.min_pattern_size), args.max_pattern_size),
            candidate_count=max(1, args.candidate_count),
            batch_size=max(1, args.batch_size),
            prompt_batch_size=max(1, args.prompt_batch_size),
            max_training_batches=max(0, args.max_training_batches),
            model=args.model,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            output=args.output,
            save_db=not args.no_save_db,
        )
    )


if __name__ == "__main__":
    main()
