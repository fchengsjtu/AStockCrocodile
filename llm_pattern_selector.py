from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd

from a_share_crawler import DEFAULT_KTYPE, mysql_connect
from stock_selector import latest_trade_date, parse_date, save_selections
from surge_pattern_miner import (
    DEFAULT_DAILY_WINDOW,
    DEFAULT_WEEKLY_WINDOW,
    SHORT_TERM_SURGE_TYPE,
    extract_features_at_position,
    iter_batches,
    load_kline_for_symbols,
    load_symbols,
)

STRATEGY_LLM_SURGE_PATTERN = "llm_surge_pattern_v1"
DEFAULT_BATCH_SIZE = 40
SELECTION_COLUMNS = [
    "TradeDate",
    "SCode",
    "SName",
    "Close",
    "Score",
    "Reason",
    "StrategyName",
    "PatternCount",
    "BestPattern",
    "BestPatternSuccessRate",
    "BestPatternFailureRate",
]


@dataclass(frozen=True)
class LlmPatternSelectionConfig:
    trade_date: str | None
    stat_type: str
    min_success_rate: float
    min_sample_count: int
    min_positive_support: int
    min_threshold: float
    train_start_date: str | None
    train_end_date: str | None
    test_start_date: str | None
    test_end_date: str | None
    daily_window: int
    weekly_window: int
    batch_size: int
    limit: int | None
    output: str | None
    save_db: bool


def normalize_rate(value: float) -> float:
    return value / 100 if value > 1 else value


def parse_pattern_text(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split("&&") if part.strip())


def failure_rate(success_rate: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(success_rate)))


def build_selection_reason(pattern_count: int, pattern_text: str, success_rate: float, sample_count: int) -> str:
    fail_rate = failure_rate(success_rate)
    reason = (
        f"patterns={pattern_count} best={pattern_text} "
        f"success={success_rate:.2%} failure={fail_rate:.2%} sample={sample_count}"
    )
    return reason[:255]


def load_surge_patterns(
    conn,
    stat_type: str,
    min_success_rate: float,
    min_sample_count: int,
    min_positive_support: int,
    min_threshold: float,
    train_start_date: date | None,
    train_end_date: date | None,
    test_start_date: date | None,
    test_end_date: date | None,
) -> pd.DataFrame:
    sql = """
        SELECT PatternText, StatType, TrainStartDate, TrainEndDate, StartDate, EndDate,
               MinSuccessRate, FeatureCount, SampleCount, SuccessCount, SuccessRate, PositiveSupport
        FROM surgepatterns
        WHERE StatType = %s
          AND SuccessRate >= %s
          AND SampleCount >= %s
          AND PositiveSupport >= %s
          AND MinSuccessRate >= %s
    """
    params: list = [stat_type, min_success_rate, min_sample_count, min_positive_support, min_threshold]
    if train_start_date is not None:
        sql += " AND TrainStartDate >= %s"
        params.append(train_start_date)
    if train_end_date is not None:
        sql += " AND TrainEndDate <= %s"
        params.append(train_end_date)
    if test_start_date is not None:
        sql += " AND StartDate >= %s"
        params.append(test_start_date)
    if test_end_date is not None:
        sql += " AND EndDate <= %s"
        params.append(test_end_date)
    sql += " ORDER BY SuccessRate DESC, SampleCount DESC, PositiveSupport DESC"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    columns = [
        "PatternText",
        "StatType",
        "TrainStartDate",
        "TrainEndDate",
        "StartDate",
        "EndDate",
        "MinSuccessRate",
        "FeatureCount",
        "SampleCount",
        "SuccessCount",
        "SuccessRate",
        "PositiveSupport",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    for column in ["TrainStartDate", "TrainEndDate", "StartDate", "EndDate"]:
        df[column] = pd.to_datetime(df[column]).dt.date
    for column in ["MinSuccessRate", "SuccessRate"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ["FeatureCount", "SampleCount", "SuccessCount", "PositiveSupport"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    df["Pattern"] = df["PatternText"].map(parse_pattern_text)
    df = df[df["Pattern"].map(bool)].copy()
    if df.empty:
        return df
    # The same pattern is stored once for each threshold bucket; keep the strongest validation row.
    df = df.sort_values(["SuccessRate", "SampleCount", "PositiveSupport"], ascending=[False, False, False])
    return df.drop_duplicates(subset=["PatternText"], keep="first").reset_index(drop=True)


def match_patterns(features: set[str], patterns: pd.DataFrame) -> pd.DataFrame:
    if patterns.empty or not features:
        return patterns.iloc[0:0].copy()
    mask = patterns["Pattern"].map(lambda pattern: set(pattern).issubset(features))
    return patterns[mask].copy()


def select_by_llm_patterns_for_date(conn, trade_date: date, config: LlmPatternSelectionConfig) -> pd.DataFrame:
    patterns = load_surge_patterns(
        conn=conn,
        stat_type=config.stat_type,
        min_success_rate=config.min_success_rate,
        min_sample_count=config.min_sample_count,
        min_positive_support=config.min_positive_support,
        min_threshold=config.min_threshold,
        train_start_date=parse_date(config.train_start_date),
        train_end_date=parse_date(config.train_end_date),
        test_start_date=parse_date(config.test_start_date),
        test_end_date=parse_date(config.test_end_date),
    )
    if patterns.empty:
        print("no surgepatterns matched the filters", flush=True)
        return pd.DataFrame(columns=SELECTION_COLUMNS)

    symbols = load_symbols(conn, trade_date, trade_date, DEFAULT_KTYPE)
    lookback_start = trade_date - timedelta(days=max(500, config.weekly_window * 10, config.daily_window * 3))
    frames = []
    batches = list(iter_batches(symbols, config.batch_size))
    print(f"select by LLM patterns trade_date={trade_date} symbols={len(symbols)} batches={len(batches)} patterns={len(patterns)}", flush=True)
    for batch_index, batch in enumerate(batches, start=1):
        daily_df = load_kline_for_symbols(conn, "dkandles", DEFAULT_KTYPE, batch, lookback_start, trade_date)
        weekly_df = load_kline_for_symbols(conn, "wkandles", "W", batch, lookback_start, trade_date)
        daily_frames = {symbol: group.sort_values("TradeDate").reset_index(drop=True) for symbol, group in daily_df.groupby("SCode", sort=False)}
        weekly_frames = {symbol: group.sort_values("TradeDate").reset_index(drop=True) for symbol, group in weekly_df.groupby("SCode", sort=False)}
        selected_rows = []
        for symbol in batch:
            daily_frame = daily_frames.get(symbol)
            weekly_frame = weekly_frames.get(symbol)
            if daily_frame is None or weekly_frame is None:
                continue
            matches = daily_frame.index[daily_frame["TradeDate"] == trade_date].tolist()
            if not matches:
                continue
            daily_pos = matches[0]
            features = extract_features_at_position(
                daily_frame,
                weekly_frame,
                daily_pos,
                trade_date,
                config.daily_window,
                config.weekly_window,
            )
            if not features:
                continue
            matched = match_patterns(features, patterns)
            if matched.empty:
                continue
            best = matched.iloc[0]
            current = daily_frame.iloc[daily_pos]
            success_rate = float(best.SuccessRate)
            fail_rate = failure_rate(success_rate)
            reason = build_selection_reason(len(matched), best.PatternText, success_rate, int(best.SampleCount))
            selected_rows.append(
                {
                    "TradeDate": trade_date,
                    "SCode": symbol,
                    "SName": current["SName"],
                    "Close": current["Close"],
                    "Score": success_rate,
                    "Reason": reason,
                    "StrategyName": STRATEGY_LLM_SURGE_PATTERN,
                    "PatternCount": len(matched),
                    "BestPattern": best.PatternText,
                    "BestPatternSuccessRate": success_rate,
                    "BestPatternFailureRate": fail_rate,
                }
            )
        if selected_rows:
            frames.append(pd.DataFrame(selected_rows))
        print(f"batch {batch_index}/{len(batches)} symbols={len(batch)} selected={len(selected_rows)}", flush=True)

    if not frames:
        return pd.DataFrame(columns=SELECTION_COLUMNS)
    selected = pd.concat(frames, ignore_index=True)
    selected = selected.sort_values(["Score", "PatternCount", "Close"], ascending=[False, False, False])
    if config.limit is not None and config.limit > 0:
        selected = selected.head(config.limit).copy()
    return selected


def run_selection(config: LlmPatternSelectionConfig) -> pd.DataFrame:
    with mysql_connect() as conn:
        trade_date = parse_date(config.trade_date) or latest_trade_date(conn, DEFAULT_KTYPE)
        selected = select_by_llm_patterns_for_date(conn, trade_date, config)
        saved = save_selections(conn, selected) if config.save_db else 0
    if config.output:
        selected.to_csv(config.output, index=False, encoding="utf-8-sig")
    print(f"trade_date={trade_date} selected={len(selected)} saved={saved}", flush=True)
    if not selected.empty:
        print(selected.to_string(index=False), flush=True)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select stocks using validated LLM/DeepSeek surge patterns")
    parser.add_argument("--date", dest="trade_date", help="Selection date, YYYYMMDD or YYYY-MM-DD; default latest daily K-line date")
    parser.add_argument("--stat-type", default=SHORT_TERM_SURGE_TYPE)
    parser.add_argument("--min-success-rate", type=float, default=0.35, help="Minimum actual test-set success rate")
    parser.add_argument("--min-sample-count", type=int, default=20)
    parser.add_argument("--min-positive-support", type=int, default=5)
    parser.add_argument("--min-threshold", type=float, default=0.25, help="Minimum saved threshold bucket")
    parser.add_argument("--train-start-date")
    parser.add_argument("--train-end-date")
    parser.add_argument("--test-start-date")
    parser.add_argument("--test-end-date")
    parser.add_argument("--daily-window", type=int, default=DEFAULT_DAILY_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WEEKLY_WINDOW)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", help="Optional CSV path")
    parser.add_argument("--no-save-db", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_selection(
        LlmPatternSelectionConfig(
            trade_date=args.trade_date,
            stat_type=args.stat_type,
            min_success_rate=normalize_rate(args.min_success_rate),
            min_sample_count=max(1, args.min_sample_count),
            min_positive_support=max(1, args.min_positive_support),
            min_threshold=normalize_rate(args.min_threshold),
            train_start_date=args.train_start_date,
            train_end_date=args.train_end_date,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            daily_window=max(2, args.daily_window),
            weekly_window=max(2, args.weekly_window),
            batch_size=max(1, args.batch_size),
            limit=args.limit,
            output=args.output,
            save_db=not args.no_save_db,
        )
    )


if __name__ == "__main__":
    main()
