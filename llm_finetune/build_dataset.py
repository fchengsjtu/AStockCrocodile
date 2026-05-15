from __future__ import annotations

import argparse
import hashlib
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_finetune.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_STAT_TYPE,
    DEFAULT_WINDOW,
    Event,
    build_messages,
    iter_batches,
    load_kline_map,
    mysql_connect,
    parse_date,
    pick_window,
    write_jsonl,
)

DEFAULT_SEED = 20260515


def stable_rank(seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in (seed, *parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def split_80_20(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda row: stable_rank(seed, row["metadata"]["scode"], row["metadata"]["anchor_date"], row["metadata"]["label"]))
    test_count = max(1, int(round(len(ordered) * 0.2))) if len(ordered) > 1 else 0
    return ordered[test_count:], ordered[:test_count]


def load_positive_events(conn, stat_type: str, start_date: date | None, end_date: date | None, limit: int | None) -> list[Event]:
    sql = """
        SELECT SCode, PrevTradeDate, GainRate
        FROM klinestatistics
        WHERE StatType = %s
          AND PrevTradeDate IS NOT NULL
    """
    params: list = [stat_type]
    if start_date:
        sql += " AND PrevTradeDate >= %s"
        params.append(start_date)
    if end_date:
        sql += " AND PrevTradeDate <= %s"
        params.append(end_date)
    sql += " ORDER BY SCode, PrevTradeDate"
    if limit and limit > 0:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [Event(str(row[0]), parse_date(row[1]), 1, float(row[2] or 0)) for row in rows]


def load_negative_events(conn, stat_type: str, start_date: date | None, end_date: date | None, limit: int, seed: int) -> list[Event]:
    params: list = [stat_type, "D"]
    filters = ""
    if start_date:
        filters += " AND dk.KTime >= %s"
        params.append(start_date)
    if end_date:
        filters += " AND dk.KTime < %s"
        params.append(end_date + timedelta(days=1))
    scan_limit = max(5000, limit * 120)
    params.append(scan_limit)
    sql = f"""
        SELECT dk.SCode, DATE(dk.KTime) AS AnchorDate
        FROM dkandles dk
        LEFT JOIN klinestatistics ks
          ON ks.SCode = dk.SCode
         AND ks.StatType = %s
         AND ks.PrevTradeDate = DATE(dk.KTime)
        WHERE dk.KType = %s
          {filters}
          AND ks.Id IS NULL
        ORDER BY dk.SCode, dk.KTime
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    events = [Event(str(row[0]), parse_date(row[1]), 0, None) for row in rows]
    random.Random(seed).shuffle(events)
    return events[:limit]


def materialize_events(conn, events: list[Event], daily_window: int, weekly_window: int, batch_size: int) -> list[dict]:
    if not events:
        return []
    start_date = min(event.anchor_date for event in events)
    end_date = max(event.anchor_date for event in events)
    lookback_start = start_date - timedelta(days=max(500, weekly_window * 10, daily_window * 4))
    samples: list[dict] = []
    symbols = sorted({event.scode for event in events})
    for batch_index, batch in enumerate(iter_batches(symbols, batch_size), start=1):
        daily_map = load_kline_map(conn, "dkandles", "D", batch, lookback_start, end_date)
        weekly_map = load_kline_map(conn, "wkandles", "W", batch, lookback_start, end_date)
        batch_events = [event for event in events if event.scode in set(batch)]
        batch_count = 0
        for event in batch_events:
            daily = pick_window(daily_map.get(event.scode, []), event.anchor_date, daily_window)
            weekly = pick_window(weekly_map.get(event.scode, []), event.anchor_date, weekly_window)
            if daily is None or weekly is None:
                continue
            samples.append(
                {
                    "messages": build_messages(event.scode, event.anchor_date, daily, weekly, event.label),
                    "metadata": {
                        "scode": event.scode,
                        "anchor_date": event.anchor_date.isoformat(),
                        "label": event.label,
                        "gain_rate": event.gain_rate,
                    },
                }
            )
            batch_count += 1
        print(f"materialized batch={batch_index} symbols={len(batch)} events={len(batch_events)} samples={batch_count}", flush=True)
    return samples


def build_dataset(
    output_dir: Path,
    stat_type: str,
    start_date: date | None,
    end_date: date | None,
    positive_limit: int | None,
    negative_ratio: float,
    seed: int,
    daily_window: int,
    weekly_window: int,
    batch_size: int,
) -> tuple[int, int]:
    with mysql_connect() as conn:
        positives = load_positive_events(conn, stat_type, start_date, end_date, positive_limit)
        negatives = load_negative_events(conn, stat_type, start_date, end_date, max(1, int(len(positives) * negative_ratio)), seed)
        print(f"loaded events positives={len(positives)} negatives={len(negatives)} stat_type={stat_type}", flush=True)
        samples = materialize_events(conn, positives + negatives, daily_window, weekly_window, batch_size)
    train_rows, test_rows = split_80_20(samples, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_jsonl(output_dir / "all.jsonl", samples)
    print(f"dataset written dir={output_dir} train={len(train_rows)} test={len(test_rows)} all={len(samples)}", flush=True)
    return len(train_rows), len(test_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 80/20 Qwen fine-tuning data from klinestatistics PrevTradeDate windows")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stat-type", default=DEFAULT_STAT_TYPE)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--daily-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--batch-size", type=int, default=30)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_dataset(
        output_dir=args.output_dir,
        stat_type=args.stat_type,
        start_date=parse_date(args.start_date) if args.start_date else None,
        end_date=parse_date(args.end_date) if args.end_date else None,
        positive_limit=args.positive_limit,
        negative_ratio=max(0.0, args.negative_ratio),
        seed=args.seed,
        daily_window=max(2, args.daily_window),
        weekly_window=max(2, args.weekly_window),
        batch_size=max(1, args.batch_size),
    )
if __name__ == "__main__":
    main()
