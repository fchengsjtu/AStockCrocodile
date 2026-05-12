from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations
from typing import Iterable

import pandas as pd

from a_share_crawler import DEFAULT_KTYPE, load_env_file, mysql_connect
from kline_statistics import SHORT_TERM_SURGE_TYPE, ensure_kline_statistics_table
from stock_selector import parse_date
from surge_pattern_miner import (
    DEFAULT_DAILY_WINDOW,
    DEFAULT_WEEKLY_WINDOW,
    SurgePatternConfig,
    backfill_kline_stat_selection_dates,
    evaluate_patterns,
    extract_features_for_date,
    iter_batches,
    load_kline_for_symbols,
    load_positive_events,
    make_frame_map,
    parse_success_rates,
    save_patterns,
)

DEFAULT_TEST_START_DATE = "20260101"
DEFAULT_TEST_END_DATE = "20260430"
DEFAULT_TRAIN_START_DATE = "20100101"
DEFAULT_SUCCESS_RATES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
DEFAULT_MODEL = "deepseek-r1-distill-qwen-14b"
DEFAULT_CANDIDATE_COUNT = 80
DEFAULT_BATCH_SIZE = 40
DEFAULT_API_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_API_KEY_ENV = "LOCAL_LLM_API_KEY"
DEFAULT_LOCAL_API_KEY = "local"
DEFAULT_MIN_PATTERN_SIZE = 3
DEFAULT_MAX_PATTERN_SIZE = 8
TRAINING_MODE_SUMMARY = "summary"
TRAINING_MODE_RAW_KLINE = "raw-kline"
DEFAULT_RAW_KLINE_WINDOW = 55
DEFAULT_RAW_SAMPLE_SIZE = 30


@dataclass(frozen=True)
class LlmPatternConfig:
    test_start_date: str
    test_end_date: str
    train_start_date: str
    train_end_date: str | None
    stat_type: str
    success_rates: tuple[float, ...]
    min_sample_count: int
    min_positive_support: int
    min_pattern_size: int
    max_pattern_size: int
    daily_window: int
    weekly_window: int
    batch_size: int
    model: str
    candidate_count: int
    top_features: int
    top_pairs: int
    training_mode: str
    raw_sample_size: int
    api_base_url: str
    api_key_env: str
    llm_response_file: str | None
    output: str | None
    save_db: bool


def pattern_config_from_llm(config: LlmPatternConfig) -> SurgePatternConfig:
    return SurgePatternConfig(
        test_start_date=config.test_start_date,
        test_end_date=config.test_end_date,
        train_start_date=config.train_start_date,
        train_end_date=config.train_end_date,
        stat_type=config.stat_type,
        min_success_rates=config.success_rates,
        min_sample_count=config.min_sample_count,
        min_positive_support=config.min_positive_support,
        max_pattern_size=config.max_pattern_size,
        daily_window=config.daily_window,
        weekly_window=config.weekly_window,
        batch_size=config.batch_size,
        output=config.output,
        save_db=config.save_db,
    )


def collect_positive_feature_summary(conn, positives: pd.DataFrame, config: LlmPatternConfig, start_date: date, end_date: date) -> tuple[Counter, Counter, int]:
    feature_counts: Counter[tuple[str, ...]] = Counter()
    symbols = sorted(positives["SCode"].dropna().unique().tolist())
    if not symbols:
        return feature_counts, Counter(), 0
    lookback_start = start_date - timedelta(days=max(500, config.weekly_window * 10, config.daily_window * 3))
    feature_rows = 0
    for batch_index, batch in enumerate(iter_batches(symbols, config.batch_size), start=1):
        daily_df = load_kline_for_symbols(conn, "dkandles", DEFAULT_KTYPE, batch, lookback_start, end_date)
        weekly_df = load_kline_for_symbols(conn, "wkandles", "W", batch, lookback_start, end_date)
        daily_frames = make_frame_map(daily_df)
        weekly_frames = make_frame_map(weekly_df)
        batch_events = positives[positives["SCode"].isin(batch)]
        batch_feature_rows = 0
        for event in batch_events.itertuples(index=False):
            daily_frame = daily_frames.get(event.SCode)
            weekly_frame = weekly_frames.get(event.SCode)
            if daily_frame is None or weekly_frame is None:
                continue
            features = extract_features_for_date(
                daily_frame,
                weekly_frame,
                event.SelectionDate,
                config.daily_window,
                config.weekly_window,
            )
            if not features:
                continue
            feature_rows += 1
            batch_feature_rows += 1
            ordered_features = sorted(features)
            for feature in ordered_features:
                feature_counts[(feature,)] += 1
            for pair in combinations(ordered_features, 2):
                feature_counts[pair] += 1
        print(
            f"llm summary batch {batch_index} symbols={len(batch)} events={len(batch_events)} "
            f"feature_rows={batch_feature_rows} candidate_patterns={len(feature_counts)}",
            flush=True,
        )

    single_counts = Counter({pattern: count for pattern, count in feature_counts.items() if len(pattern) == 1})
    pair_counts = Counter({pattern: count for pattern, count in feature_counts.items() if len(pattern) == 2})
    return single_counts, pair_counts, feature_rows


def _format_counter(counter: Counter, total: int, limit: int) -> list[dict]:
    rows = []
    for pattern, count in counter.most_common(limit):
        rows.append(
            {
                "features": list(pattern),
                "positive_support": int(count),
                "positive_rate": round(count / total, 6) if total else 0.0,
            }
        )
    return rows


def _round_value(value) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 4)


def _frame_window_to_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    columns = ["TradeDate", "Open", "Close", "High", "Low", "Volume", "Amount", "MA5", "MA13", "MA34", "MA55"]
    for item in frame[columns].itertuples(index=False):
        records.append(
            {
                "date": str(item.TradeDate),
                "open": _round_value(item.Open),
                "close": _round_value(item.Close),
                "high": _round_value(item.High),
                "low": _round_value(item.Low),
                "volume": _round_value(item.Volume),
                "amount": _round_value(item.Amount),
                "ma5": _round_value(item.MA5),
                "ma13": _round_value(item.MA13),
                "ma34": _round_value(item.MA34),
                "ma55": _round_value(item.MA55),
            }
        )
    return records


def _select_evenly_spaced_events(positives: pd.DataFrame, limit: int) -> pd.DataFrame:
    if positives.empty or len(positives) <= limit:
        return positives.copy()
    ordered = positives.sort_values(["SelectionDate", "SCode"]).reset_index(drop=True)
    indexes = sorted({round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)})
    return ordered.iloc[indexes].reset_index(drop=True)


def collect_positive_kline_samples(conn, positives: pd.DataFrame, config: LlmPatternConfig, start_date: date, end_date: date) -> tuple[list[dict], Counter, int]:
    samples = []
    feature_counts: Counter[tuple[str, ...]] = Counter()
    target_events = _select_evenly_spaced_events(positives, config.raw_sample_size)
    symbols = sorted(target_events["SCode"].dropna().unique().tolist())
    if not symbols:
        return samples, feature_counts, 0
    lookback_start = start_date - timedelta(days=max(500, config.weekly_window * 10, config.daily_window * 3))
    for batch_index, batch in enumerate(iter_batches(symbols, config.batch_size), start=1):
        daily_df = load_kline_for_symbols(conn, "dkandles", DEFAULT_KTYPE, batch, lookback_start, end_date)
        weekly_df = load_kline_for_symbols(conn, "wkandles", "W", batch, lookback_start, end_date)
        daily_frames = make_frame_map(daily_df)
        weekly_frames = make_frame_map(weekly_df)
        batch_events = target_events[target_events["SCode"].isin(batch)]
        batch_samples = 0
        for event in batch_events.itertuples(index=False):
            daily_frame = daily_frames.get(event.SCode)
            weekly_frame = weekly_frames.get(event.SCode)
            if daily_frame is None or weekly_frame is None:
                continue
            daily_matches = daily_frame.index[daily_frame["TradeDate"] == event.SelectionDate].tolist()
            if not daily_matches:
                continue
            daily_pos = daily_matches[0]
            weekly_positions = weekly_frame.index[weekly_frame["TradeDate"] <= event.SelectionDate].tolist()
            if not weekly_positions:
                continue
            weekly_pos = weekly_positions[-1]
            if daily_pos + 1 < config.daily_window or weekly_pos + 1 < config.weekly_window:
                continue
            daily_window = daily_frame.iloc[daily_pos + 1 - config.daily_window : daily_pos + 1]
            weekly_window = weekly_frame.iloc[weekly_pos + 1 - config.weekly_window : weekly_pos + 1]
            features = extract_features_for_date(daily_frame, weekly_frame, event.SelectionDate, config.daily_window, config.weekly_window)
            if not features:
                continue
            for feature in sorted(features):
                feature_counts[(feature,)] += 1
            samples.append(
                {
                    "scode": event.SCode,
                    "sname": event.SName,
                    "selection_date": str(event.SelectionDate),
                    "start_rise_date": str(event.StartRiseDate),
                    "gain_rate": _round_value(event.GainRate),
                    "daily_bars": _frame_window_to_records(daily_window),
                    "weekly_bars": _frame_window_to_records(weekly_window),
                }
            )
            batch_samples += 1
        print(
            f"llm raw-kline batch {batch_index} symbols={len(batch)} events={len(batch_events)} samples={batch_samples}",
            flush=True,
        )
    return samples, feature_counts, len(samples)


def build_llm_prompt(
    single_counts: Counter,
    pair_counts: Counter,
    feature_rows: int,
    config: LlmPatternConfig,
    train_start_date: date,
    train_end_date: date,
    test_start_date: date,
    test_end_date: date,
) -> str:
    payload = {
        "task": "Generate candidate technical setup patterns for A-share short-term surge mining.",
        "positive_label": config.stat_type,
        "train_range": [str(train_start_date), str(train_end_date)],
        "test_range_for_validation": [str(test_start_date), str(test_end_date)],
        "windows": {
            "daily": f"SelectionDate plus previous {config.daily_window - 1} daily bars",
            "weekly": f"SelectionDate plus previous {config.weekly_window - 1} weekly bars",
        },
        "positive_feature_rows": feature_rows,
        "min_features_per_pattern": config.min_pattern_size,
        "max_features_per_pattern": config.max_pattern_size,
        "candidate_count": config.candidate_count,
        "top_single_features": _format_counter(single_counts, feature_rows, config.top_features),
        "top_pair_features": _format_counter(pair_counts, feature_rows, config.top_pairs),
    }
    return (
        "You are a quantitative trading research assistant. "
        "Propose diverse, testable pattern candidates using ONLY exact feature tokens present in the JSON. "
        "Prefer robust combinations that could generalize, not merely the highest-support pairs. "
        "Return strict JSON only, with this schema: "
        '{"patterns":[{"name":"short_name","features":["TOKEN_A","TOKEN_B","TOKEN_C"],"rationale":"brief reason"}]}. '
        "Do not include markdown or extra commentary.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def build_raw_kline_llm_prompt(
    samples: list[dict],
    feature_counts: Counter,
    config: LlmPatternConfig,
    train_start_date: date,
    train_end_date: date,
    test_start_date: date,
    test_end_date: date,
) -> str:
    valid_features = sorted(feature[0] for feature in feature_counts if len(feature) == 1)
    payload = {
        "task": "Generate candidate technical setup patterns from raw K-line positive samples.",
        "positive_label": config.stat_type,
        "train_range": [str(train_start_date), str(train_end_date)],
        "test_range_for_validation": [str(test_start_date), str(test_end_date)],
        "input_mode": "raw-kline",
        "windows": {
            "daily": f"{config.daily_window} daily bars ending at SelectionDate, SelectionDate included",
            "weekly": f"{config.weekly_window} weekly bars ending at SelectionDate, latest weekly bar on or before SelectionDate included",
        },
        "candidate_count": config.candidate_count,
        "min_features_per_pattern": config.min_pattern_size,
        "max_features_per_pattern": config.max_pattern_size,
        "allowed_feature_tokens": valid_features,
        "positive_samples": samples,
    }
    return (
        "You are a quantitative trading research assistant. "
        "Study ONLY the raw daily_bars and weekly_bars in the positive samples, then propose diverse setup patterns. "
        "For validation compatibility, every returned feature must be chosen exactly from allowed_feature_tokens. "
        "Return strict JSON only, with this schema: "
        '{"patterns":[{"name":"short_name","features":["TOKEN_A","TOKEN_B","TOKEN_C"],"rationale":"brief reason from raw K-line windows"}]}. '
        "Do not include markdown or extra commentary.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def call_llm_chat(prompt: str, config: LlmPatternConfig) -> str:
    load_env_file()
    api_key = os.environ.get(config.api_key_env) or DEFAULT_LOCAL_API_KEY
    timeout_seconds = int(os.environ.get("LOCAL_LLM_TIMEOUT", "600"))
    max_tokens = int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "2048"))
    url = config.api_base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": "You generate strict JSON candidate rule sets for quantitative validation.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if os.environ.get("LOCAL_LLM_RESPONSE_FORMAT", "0").lower() not in {"0", "false", "no", "off"}:
        body["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code in {400, 404, 422} and "response_format" in body:
            body.pop("response_format", None)
            try:
                request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as retry_exc:
                retry_detail = retry_exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"LLM request failed after compatibility retry: HTTP {retry_exc.code}: {retry_detail}") from retry_exc
        raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {detail}") from exc
    return payload["choices"][0]["message"]["content"]


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    if not stripped.startswith("{") or not stripped.endswith("}"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def parse_llm_patterns(response_text: str, valid_features: set[str], min_pattern_size: int, max_pattern_size: int) -> list[tuple[str, ...]]:
    payload = extract_json_object(response_text)
    patterns = []
    seen = set()
    for item in payload.get("patterns", []):
        raw_features = item.get("features", [])
        if not raw_features and isinstance(item.get("pattern"), str):
            raw_features = [item["pattern"]]
        if not raw_features and isinstance(item.get("PatternText"), str):
            raw_features = [item["PatternText"]]
        expanded_features = []
        for feature in raw_features:
            if not isinstance(feature, str):
                continue
            expanded_features.extend(part.strip() for part in feature.split("&&") if part.strip())
        features = tuple(feature for feature in expanded_features if feature in valid_features)
        if len(features) < min_pattern_size or len(features) > max_pattern_size:
            continue
        features = tuple(sorted(set(features)))
        if len(features) != len(expanded_features):
            continue
        if features not in seen:
            patterns.append(features)
            seen.add(features)
    return patterns


def fallback_patterns_from_counts(feature_counts: Counter, min_pattern_size: int, max_pattern_size: int, limit: int) -> list[tuple[str, ...]]:
    patterns = [
        pattern
        for pattern, _ in feature_counts.most_common()
        if min_pattern_size <= len(pattern) <= max_pattern_size
    ]
    if patterns:
        return patterns[:limit]
    top_features = [pattern[0] for pattern, _ in feature_counts.most_common() if len(pattern) == 1]
    fallback = []
    for size in range(min_pattern_size, max_pattern_size + 1):
        for pattern in combinations(top_features, size):
            fallback.append(tuple(sorted(pattern)))
            if len(fallback) >= limit:
                return fallback
    return fallback


def estimate_positive_support(pattern: tuple[str, ...], single_counts: Counter, pair_counts: Counter) -> int:
    if pattern in pair_counts:
        return int(pair_counts[pattern])
    supports = [int(single_counts.get((feature,), 0)) for feature in pattern]
    return min(supports) if supports else 0


def load_or_generate_llm_patterns(
    single_counts: Counter,
    pair_counts: Counter,
    feature_rows: int,
    config: LlmPatternConfig,
    train_start_date: date,
    train_end_date: date,
    test_start_date: date,
    test_end_date: date,
) -> list[tuple[str, ...]]:
    valid_features = {pattern[0] for pattern in single_counts}
    prompt = build_llm_prompt(
        single_counts,
        pair_counts,
        feature_rows,
        config,
        train_start_date,
        train_end_date,
        test_start_date,
        test_end_date,
    )
    if config.llm_response_file:
        with open(config.llm_response_file, "r", encoding="utf-8-sig") as file:
            response_text = file.read()
    else:
            response_text = call_llm_chat(prompt, config)
    patterns = parse_llm_patterns(response_text, valid_features, config.min_pattern_size, config.max_pattern_size)
    if not patterns:
        raise RuntimeError("LLM did not return any valid patterns using known feature tokens")
    print(f"llm candidate patterns={len(patterns)}", flush=True)
    return patterns[: config.candidate_count]


def load_or_generate_raw_kline_patterns(
    samples: list[dict],
    feature_counts: Counter,
    config: LlmPatternConfig,
    train_start_date: date,
    train_end_date: date,
    test_start_date: date,
    test_end_date: date,
) -> list[tuple[str, ...]]:
    valid_features = {feature[0] for feature in feature_counts if len(feature) == 1}
    prompt = build_raw_kline_llm_prompt(
        samples,
        feature_counts,
        config,
        train_start_date,
        train_end_date,
        test_start_date,
        test_end_date,
    )
    if config.llm_response_file:
        with open(config.llm_response_file, "r", encoding="utf-8-sig") as file:
            response_text = file.read()
    else:
        response_text = call_llm_chat(prompt, config)
    patterns = parse_llm_patterns(response_text, valid_features, config.min_pattern_size, config.max_pattern_size)
    if not patterns:
        patterns = fallback_patterns_from_counts(feature_counts, config.min_pattern_size, config.max_pattern_size, config.candidate_count)
        if not patterns:
            raise RuntimeError("LLM did not return any valid raw-kline patterns using known feature tokens")
        print(f"LLM returned no valid raw-kline patterns; using {len(patterns)} high-support fallback patterns", flush=True)
    print(f"llm raw-kline candidate patterns={len(patterns)}", flush=True)
    return patterns[: config.candidate_count]


def run_llm_pattern_mining(config: LlmPatternConfig) -> pd.DataFrame:
    test_start_date = parse_date(config.test_start_date)
    test_end_date = parse_date(config.test_end_date)
    train_start_date = parse_date(config.train_start_date)
    train_end_date = parse_date(config.train_end_date) if config.train_end_date else None
    if test_start_date is None or test_end_date is None or train_start_date is None:
        raise ValueError("test-start-date, test-end-date, and train-start-date are required")
    if train_end_date is None:
        train_end_date = test_start_date - timedelta(days=1)
    if train_start_date > train_end_date:
        raise ValueError("train-start-date must be <= train-end-date")
    if test_start_date > test_end_date:
        raise ValueError("test-start-date must be <= test-end-date")

    base_config = pattern_config_from_llm(config)
    with mysql_connect() as conn:
        ensure_kline_statistics_table(conn)
        if config.save_db:
            from surge_pattern_miner import ensure_surge_pattern_table

            ensure_surge_pattern_table(conn)
        backfilled = backfill_kline_stat_selection_dates(conn, config.stat_type)
        if backfilled:
            print(f"backfilled SelectionDate rows={backfilled}", flush=True)
        train_positives = load_positive_events(conn, config.stat_type, train_start_date, train_end_date)
        test_positives = load_positive_events(conn, config.stat_type, test_start_date, test_end_date)
        print(
            f"loaded positives stat_type={config.stat_type} "
            f"train_rows={len(train_positives)} test_rows={len(test_positives)}",
            flush=True,
        )
        if test_positives.empty:
            print(
                "WARNING test positives are empty. Generate test samples first, for example: "
                "python ./kline_statistics.py --start-date 20260101 --end-date 20260430",
                flush=True,
            )
        if config.training_mode == TRAINING_MODE_RAW_KLINE:
            samples, feature_counts, feature_rows = collect_positive_kline_samples(
                conn,
                train_positives,
                config,
                train_start_date,
                train_end_date,
            )
            llm_patterns = load_or_generate_raw_kline_patterns(
                samples,
                feature_counts,
                config,
                train_start_date,
                train_end_date,
                test_start_date,
                test_end_date,
            )
            single_counts = feature_counts
            pair_counts = Counter()
        else:
            single_counts, pair_counts, feature_rows = collect_positive_feature_summary(
                conn,
                train_positives,
                config,
                train_start_date,
                train_end_date,
            )
            llm_patterns = load_or_generate_llm_patterns(
                single_counts,
                pair_counts,
                feature_rows,
                config,
                train_start_date,
                train_end_date,
                test_start_date,
                test_end_date,
            )
        positive_support = Counter()
        for pattern in llm_patterns:
            positive_support[pattern] = estimate_positive_support(pattern, single_counts, pair_counts)
        patterns = evaluate_patterns(
            conn,
            test_positives,
            set(llm_patterns),
            positive_support,
            base_config,
            test_start_date,
            test_end_date,
        )
        if config.output:
            patterns.to_csv(config.output, index=False, encoding="utf-8-sig")
        saved = (
            save_patterns(conn, patterns, base_config, train_start_date, train_end_date, test_start_date, test_end_date)
            if config.save_db
            else 0
        )
    print(
        f"llm surge patterns kept={len(patterns)} saved={saved} "
        f"success_rates={','.join(f'{rate:.0%}' for rate in config.success_rates)}",
        flush=True,
    )
    return patterns


def build_parser() -> argparse.ArgumentParser:
    load_env_file()
    parser = argparse.ArgumentParser(description="Use a local OpenAI-compatible LLM to propose surge setup patterns, then validate them on historical K-lines")
    parser.add_argument("--test-start-date", default=DEFAULT_TEST_START_DATE, help="Test selection date start")
    parser.add_argument("--test-end-date", default=DEFAULT_TEST_END_DATE, help="Test selection date end")
    parser.add_argument("--train-start-date", default=DEFAULT_TRAIN_START_DATE, help="Training selection date start")
    parser.add_argument("--train-end-date", help="Training selection date end; default is the day before test-start-date")
    parser.add_argument("--stat-type", default=SHORT_TERM_SURGE_TYPE)
    parser.add_argument("--success-rates", type=parse_success_rates, default=DEFAULT_SUCCESS_RATES)
    parser.add_argument("--min-sample-count", type=int, default=20)
    parser.add_argument("--min-positive-support", type=int, default=5)
    parser.add_argument("--min-pattern-size", type=int, default=DEFAULT_MIN_PATTERN_SIZE, help="Minimum number of feature clauses in each LLM pattern")
    parser.add_argument("--max-pattern-size", type=int, default=DEFAULT_MAX_PATTERN_SIZE, help="Maximum number of feature clauses in each LLM pattern")
    parser.add_argument("--training-mode", choices=(TRAINING_MODE_SUMMARY, TRAINING_MODE_RAW_KLINE), default=TRAINING_MODE_SUMMARY, help="summary keeps the old feature-summary mode; raw-kline sends raw 55 daily/weekly bars to the local LLM")
    parser.add_argument("--daily-window", type=int)
    parser.add_argument("--weekly-window", type=int)
    parser.add_argument("--raw-sample-size", type=int, default=DEFAULT_RAW_SAMPLE_SIZE, help="Number of positive raw K-line samples sent to the local LLM in raw-kline mode")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model", default=os.environ.get("LOCAL_LLM_MODEL") or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--top-features", type=int, default=80)
    parser.add_argument("--top-pairs", type=int, default=120)
    parser.add_argument("--api-base-url", default=os.environ.get("LOCAL_LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_API_BASE_URL))
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--llm-response-file", help="Use a saved JSON response instead of calling the LLM API")
    parser.add_argument("--output", help="Optional CSV path for retained patterns")
    parser.add_argument("--no-save-db", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    default_window = DEFAULT_RAW_KLINE_WINDOW if args.training_mode == TRAINING_MODE_RAW_KLINE else DEFAULT_DAILY_WINDOW
    daily_window = args.daily_window or default_window
    weekly_window = args.weekly_window or default_window
    run_llm_pattern_mining(
        LlmPatternConfig(
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            train_start_date=args.train_start_date,
            train_end_date=args.train_end_date,
            stat_type=args.stat_type,
            success_rates=args.success_rates,
            min_sample_count=max(1, args.min_sample_count),
            min_positive_support=max(1, args.min_positive_support),
            min_pattern_size=max(1, args.min_pattern_size),
            max_pattern_size=max(max(1, args.min_pattern_size), args.max_pattern_size),
            daily_window=max(2, daily_window),
            weekly_window=max(2, weekly_window),
            batch_size=max(1, args.batch_size),
            model=args.model,
            candidate_count=max(1, args.candidate_count),
            top_features=max(1, args.top_features),
            top_pairs=max(1, args.top_pairs),
            training_mode=args.training_mode,
            raw_sample_size=max(1, args.raw_sample_size),
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            llm_response_file=args.llm_response_file,
            output=args.output,
            save_db=not args.no_save_db,
        )
    )


if __name__ == "__main__":
    main()
