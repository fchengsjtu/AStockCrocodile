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
    iter_pattern_keys,
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
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_CANDIDATE_COUNT = 80
DEFAULT_BATCH_SIZE = 40


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
    max_pattern_size: int
    daily_window: int
    weekly_window: int
    batch_size: int
    model: str
    candidate_count: int
    top_features: int
    top_pairs: int
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
            for pattern in iter_pattern_keys(features, config.max_pattern_size):
                feature_counts[pattern] += 1
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
        '{"patterns":[{"name":"short_name","features":["TOKEN_A","TOKEN_B"],"rationale":"brief reason"}]}. '
        "Do not include markdown or extra commentary.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def call_openai_compatible_chat(prompt: str, config: LlmPatternConfig) -> str:
    load_env_file()
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing {config.api_key_env}; set it in env.txt or the process environment")
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
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {detail}") from exc
    return payload["choices"][0]["message"]["content"]


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def parse_llm_patterns(response_text: str, valid_features: set[str], max_pattern_size: int) -> list[tuple[str, ...]]:
    payload = extract_json_object(response_text)
    patterns = []
    seen = set()
    for item in payload.get("patterns", []):
        raw_features = item.get("features", [])
        features = tuple(feature for feature in raw_features if isinstance(feature, str) and feature in valid_features)
        if not features or len(features) > max_pattern_size:
            continue
        features = tuple(sorted(set(features)))
        if len(features) != len(raw_features):
            continue
        if features not in seen:
            patterns.append(features)
            seen.add(features)
    return patterns


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
        response_text = call_openai_compatible_chat(prompt, config)
    patterns = parse_llm_patterns(response_text, valid_features, config.max_pattern_size)
    if not patterns:
        raise RuntimeError("LLM did not return any valid patterns using known feature tokens")
    print(f"llm candidate patterns={len(patterns)}", flush=True)
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
            positive_support[pattern] = single_counts.get(pattern, pair_counts.get(pattern, 0))
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
    parser = argparse.ArgumentParser(description="Use an LLM to propose surge setup patterns, then validate them on historical K-lines")
    parser.add_argument("--test-start-date", default=DEFAULT_TEST_START_DATE, help="Test selection date start")
    parser.add_argument("--test-end-date", default=DEFAULT_TEST_END_DATE, help="Test selection date end")
    parser.add_argument("--train-start-date", default=DEFAULT_TRAIN_START_DATE, help="Training selection date start")
    parser.add_argument("--train-end-date", help="Training selection date end; default is the day before test-start-date")
    parser.add_argument("--stat-type", default=SHORT_TERM_SURGE_TYPE)
    parser.add_argument("--success-rates", type=parse_success_rates, default=DEFAULT_SUCCESS_RATES)
    parser.add_argument("--min-sample-count", type=int, default=20)
    parser.add_argument("--min-positive-support", type=int, default=5)
    parser.add_argument("--max-pattern-size", type=int, default=2)
    parser.add_argument("--daily-window", type=int, default=DEFAULT_DAILY_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WEEKLY_WINDOW)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--top-features", type=int, default=80)
    parser.add_argument("--top-pairs", type=int, default=120)
    parser.add_argument("--api-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-response-file", help="Use a saved JSON response instead of calling the LLM API")
    parser.add_argument("--output", help="Optional CSV path for retained patterns")
    parser.add_argument("--no-save-db", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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
            max_pattern_size=max(1, args.max_pattern_size),
            daily_window=max(2, args.daily_window),
            weekly_window=max(2, args.weekly_window),
            batch_size=max(1, args.batch_size),
            model=args.model,
            candidate_count=max(1, args.candidate_count),
            top_features=max(1, args.top_features),
            top_pairs=max(1, args.top_pairs),
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            llm_response_file=args.llm_response_file,
            output=args.output,
            save_db=not args.no_save_db,
        )
    )


if __name__ == "__main__":
    main()
