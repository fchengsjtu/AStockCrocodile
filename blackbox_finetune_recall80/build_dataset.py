from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune.build_dataset import (
    excluded_dates_by_symbol,
    load_excluded_positive_windows,
    load_positive_events,
    split_train_test,
)
from blackbox_finetune_recall80.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_STAT_TYPE,
    DEFAULT_TRAIN_END_DATE,
    DEFAULT_TRAIN_START_DATE,
    DEFAULT_SAMPLE_MODE,
    sample_mode_config,
    SampleEvent,
    materialize_events,
    mysql_connect,
    parse_date,
    write_jsonl,
)

DEFAULT_SEED = 20260518


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


def build_recall80_dataset(
    output_dir: Path,
    stat_type,
    start_date,
    end_date,
    positive_limit: int | None,
    negative_ratio: float,
    train_ratio: float,
    seed: int,
    daily_window: int | None,
    weekly_window: int | None,
    monthly_window: int | None,
    batch_size: int,
    sample_mode: str,
) -> tuple[int, int]:
    with mysql_connect() as conn:
        positives = load_positive_events(conn, stat_type, start_date, end_date, positive_limit)
        negative_limit = max(1, int(len(positives) * negative_ratio))
        negatives = load_random_negative_events(conn, stat_type, start_date, end_date, negative_limit, seed, batch_size)
        all_events = positives + negatives
        print(f"loaded events positives={len(positives)} negatives={len(negatives)}", flush=True)
        samples = materialize_events(conn, all_events, daily_window, weekly_window, batch_size, sample_mode=sample_mode, monthly_window=monthly_window)
    train_rows, test_rows = split_train_test(samples, train_ratio, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_jsonl(output_dir / "all.jsonl", samples)
    print(f"dataset written dir={output_dir} train={len(train_rows)} test={len(test_rows)} all={len(samples)}", flush=True)
    return len(train_rows), len(test_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build recall80 black-box fine-tuning training samples")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stat-type", default=DEFAULT_STAT_TYPE)
    parser.add_argument("--start-date", default=os.environ.get("TRAIN_START_DATE", DEFAULT_TRAIN_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("TRAIN_END_DATE", DEFAULT_TRAIN_END_DATE))
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--negative-ratio", type=float, default=float(os.environ.get("NEGATIVE_RATIO", "3.0")))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-mode", choices=["short", "long", "xlong"], default=DEFAULT_SAMPLE_MODE)
    parser.add_argument("--daily-window", type=int, help="Override daily bars for the selected sample mode")
    parser.add_argument("--weekly-window", type=int, help="Override weekly bars for the selected sample mode")
    parser.add_argument("--monthly-window", type=int, help="Override monthly bars for the selected sample mode")
    parser.add_argument("--batch-size", type=int, default=80)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_recall80_dataset(
        output_dir=args.output_dir,
        stat_type=args.stat_type,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        positive_limit=args.positive_limit,
        negative_ratio=max(0.0, args.negative_ratio),
        train_ratio=min(max(args.train_ratio, 0.01), 0.99),
        seed=args.seed,
        daily_window=max(2, args.daily_window) if args.daily_window else None,
        weekly_window=max(2, args.weekly_window) if args.weekly_window else None,
        monthly_window=max(0, args.monthly_window) if args.monthly_window is not None else None,
        batch_size=max(1, args.batch_size),
        sample_mode=args.sample_mode,
    )


if __name__ == "__main__":
    main()
