from __future__ import annotations

import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_STAT_TYPE,
    DEFAULT_WINDOW,
    SampleEvent,
    event_key,
    materialize_events,
    mysql_connect,
    parse_date,
    write_jsonl,
)
from goal_pattern_search import stable_rank

DEFAULT_SEED = 20260517


def load_positive_events(conn, stat_type: str, start_date: date, end_date: date, limit: int | None = None) -> list[SampleEvent]:
    sql = """
        SELECT SCode, PrevTradeDate, GainRate
        FROM klinestatistics
        WHERE StatType = %s
          AND PrevTradeDate >= %s
          AND PrevTradeDate <= %s
          AND PrevTradeDate IS NOT NULL
        ORDER BY SCode, PrevTradeDate
    """
    params: list = [stat_type, start_date, end_date]
    if limit and limit > 0:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [SampleEvent(str(row[0]), parse_date(row[1]), 1, "positive", float(row[2] or 0)) for row in rows]


def load_excluded_positive_windows(conn, stat_type: str, start_date: date, end_date: date, padding_days: int = 10) -> dict[str, set[date]]:
    sql = """
        SELECT SCode, PrevTradeDate
        FROM klinestatistics
        WHERE StatType = %s
          AND PrevTradeDate >= %s
          AND PrevTradeDate <= %s
          AND PrevTradeDate IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, (stat_type, start_date - timedelta(days=padding_days), end_date + timedelta(days=padding_days)))
        rows = cur.fetchall()
    raw_dates: dict[str, set[date]] = {}
    for scode, prev_trade_date in rows:
        raw_dates.setdefault(str(scode), set()).add(parse_date(prev_trade_date))
    return raw_dates


def load_trading_dates_for_symbols(conn, symbols: list[str], start_date: date, end_date: date) -> dict[str, list[date]]:
    if not symbols:
        return {}
    placeholders = ",".join(["%s"] * len(symbols))
    sql = f"""
        SELECT SCode, DATE(KTime)
        FROM dkandles
        WHERE KType = 'D'
          AND SCode IN ({placeholders})
          AND KTime >= %s
          AND KTime < %s
        ORDER BY SCode, KTime
    """
    with conn.cursor() as cur:
        cur.execute(sql, [*symbols, start_date, end_date + timedelta(days=1)])
        rows = cur.fetchall()
    dates: dict[str, list[date]] = {}
    for scode, trade_date in rows:
        dates.setdefault(str(scode), []).append(parse_date(trade_date))
    return dates


def excluded_dates_by_symbol(conn, positive_windows: dict[str, set[date]], start_date: date, end_date: date, buffer: int, batch_size: int) -> dict[str, set[date]]:
    result: dict[str, set[date]] = {scode: set() for scode in positive_windows}
    symbols = sorted(positive_windows)
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        date_map = load_trading_dates_for_symbols(conn, batch, start_date - timedelta(days=20), end_date + timedelta(days=20))
        for scode in batch:
            dates = date_map.get(scode, [])
            index = {trade_date: idx for idx, trade_date in enumerate(dates)}
            for pos_date in positive_windows.get(scode, set()):
                idx = index.get(pos_date)
                if idx is None:
                    continue
                lo = max(0, idx - buffer)
                hi = min(len(dates), idx + buffer + 1)
                result.setdefault(scode, set()).update(dates[lo:hi])
    return result


def load_negative_events(
    conn,
    stat_type: str,
    start_date: date,
    end_date: date,
    limit: int,
    seed: int,
    batch_size: int,
) -> list[SampleEvent]:
    positive_windows = load_excluded_positive_windows(conn, stat_type, start_date, end_date)
    excluded = excluded_dates_by_symbol(conn, positive_windows, start_date, end_date, 3, batch_size)
    scan_limit = max(limit * 30, 5000)
    sql = """
        SELECT SCode, DATE(KTime)
        FROM dkandles
        WHERE KType = 'D'
          AND KTime >= %s
          AND KTime < %s
        ORDER BY SCode, KTime
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (start_date, end_date + timedelta(days=1), scan_limit))
        rows = cur.fetchall()
    candidates: list[SampleEvent] = []
    for scode, trade_date_value in rows:
        scode = str(scode)
        trade_date = parse_date(trade_date_value)
        if trade_date in excluded.get(scode, set()):
            continue
        candidates.append(SampleEvent(scode, trade_date, 0, "negative", None))
    candidates = sorted(candidates, key=lambda event: stable_rank(seed, event.scode, event.anchor_date, "negative"))
    return candidates[:limit]


def load_random_negative_events(
    conn,
    stat_type: str,
    start_date: date,
    end_date: date,
    limit: int,
    seed: int,
    batch_size: int,
) -> list[SampleEvent]:
    positive_windows = load_excluded_positive_windows(conn, stat_type, start_date, end_date)
    excluded = excluded_dates_by_symbol(conn, positive_windows, start_date, end_date, 3, batch_size)
    candidates: list[SampleEvent] = []
    seen: set[tuple[str, date]] = set()
    scan_limit = max(limit * 3, limit + 1000, 5000)
    attempt = 0
    while len(candidates) < limit and attempt < 5:
        sql = """
            SELECT SCode, DATE(KTime)
            FROM dkandles
            WHERE KType = 'D'
              AND KTime >= %s
              AND KTime < %s
            ORDER BY RAND(%s)
            LIMIT %s
        """
        with conn.cursor() as cur:
            cur.execute(sql, (start_date, end_date + timedelta(days=1), seed + attempt, scan_limit))
            rows = cur.fetchall()
        for scode, trade_date_value in rows:
            scode = str(scode)
            trade_date = parse_date(trade_date_value)
            key = (scode, trade_date)
            if key in seen or trade_date in excluded.get(scode, set()):
                continue
            seen.add(key)
            candidates.append(SampleEvent(scode, trade_date, 0, "negative", None))
            if len(candidates) >= limit:
                break
        attempt += 1
        scan_limit *= 2
    return candidates[:limit]


def split_train_test(rows: list[dict], train_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda row: stable_rank(seed, row["metadata"]["scode"], row["metadata"]["anchor_date"], row["metadata"]["label"]))
    test_count = max(1, int(round(len(ordered) * (1.0 - train_ratio)))) if len(ordered) > 1 else 0
    return ordered[test_count:], ordered[:test_count]


def build_dataset(
    output_dir: Path,
    stat_type: str,
    start_date: date,
    end_date: date,
    positive_limit: int | None,
    negative_ratio: float,
    train_ratio: float,
    seed: int,
    daily_window: int,
    weekly_window: int,
    batch_size: int,
) -> tuple[int, int]:
    with mysql_connect() as conn:
        positives = load_positive_events(conn, stat_type, start_date, end_date, positive_limit)
        negative_limit = max(1, int(len(positives) * negative_ratio))
        negatives = load_negative_events(conn, stat_type, start_date, end_date, negative_limit, seed, batch_size)
        all_events = positives + negatives
        print(f"loaded events positives={len(positives)} negatives={len(negatives)}", flush=True)
        samples = materialize_events(conn, all_events, daily_window, weekly_window, batch_size)
    train_rows, test_rows = split_train_test(samples, train_ratio, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_jsonl(output_dir / "all.jsonl", samples)
    print(f"dataset written dir={output_dir} train={len(train_rows)} test={len(test_rows)} all={len(samples)}", flush=True)
    return len(train_rows), len(test_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build black-box Qwen fine-tuning samples with positive windows excluded from negatives")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stat-type", default=DEFAULT_STAT_TYPE)
    parser.add_argument("--start-date", default="20110101")
    parser.add_argument("--end-date", default="20241231")
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--daily-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--batch-size", type=int, default=80)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_dataset(
        output_dir=args.output_dir,
        stat_type=args.stat_type,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        positive_limit=args.positive_limit,
        negative_ratio=max(0.0, args.negative_ratio),
        train_ratio=min(max(args.train_ratio, 0.01), 0.99),
        seed=args.seed,
        daily_window=max(2, args.daily_window),
        weekly_window=max(2, args.weekly_window),
        batch_size=max(1, args.batch_size),
    )


if __name__ == "__main__":
    main()
