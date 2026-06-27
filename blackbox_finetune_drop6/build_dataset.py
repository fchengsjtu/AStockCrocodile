from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_drop6.common import (
    DEFAULT_STAT_TYPE,
    DEFAULT_TRAIN_END_DATE,
    DEFAULT_TRAIN_SEED,
    DEFAULT_TRAIN_START_DATE,
    DEFAULT_SAMPLE_MODE,
    SampleEvent,
    default_data_dir,
    materialize_events,
    mysql_connect,
    parse_date,
    write_jsonl,
)

DROP_LOOKAHEAD_TRADING_DAYS = 3
DROP_THRESHOLD = 0.06
NEGATIVE_EXCLUDE_TRADING_DAYS = 10


@dataclass(frozen=True)
class DailyBar:
    scode: str
    trade_date: date
    close: float
    low: float


def _to_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _load_daily_bars_by_symbol(conn, start_date: date, end_date: date) -> dict[str, list[DailyBar]]:
    query_start = start_date - timedelta(days=45)
    query_end = end_date + timedelta(days=20)
    rows_by_symbol: dict[str, list[DailyBar]] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT SCode, DATE(KTime), Close, Low
            FROM dkandles
            WHERE KType = 'D'
              AND KTime >= %s
              AND KTime < %s
              AND Close IS NOT NULL
              AND Low IS NOT NULL
            ORDER BY SCode, KTime
            """,
            (query_start, query_end + timedelta(days=1)),
        )
        for scode, trade_date, close_value, low_value in cur.fetchall():
            close = _to_float(close_value)
            low = _to_float(low_value)
            if close is None or low is None:
                continue
            rows_by_symbol.setdefault(str(scode), []).append(
                DailyBar(str(scode), parse_date(trade_date), close, low)
            )
    return rows_by_symbol


def _build_events_for_symbol(
    rows: list[DailyBar],
    start_date: date,
    end_date: date,
    *,
    drop_threshold: float = DROP_THRESHOLD,
    lookahead_days: int = DROP_LOOKAHEAD_TRADING_DAYS,
    negative_exclude_days: int = NEGATIVE_EXCLUDE_TRADING_DAYS,
) -> tuple[list[SampleEvent], list[SampleEvent]]:
    positives: list[SampleEvent] = []
    negative_candidates: list[SampleEvent] = []
    excluded_negative_indices: set[int] = set()
    anchor_indices: list[int] = []

    for index, row in enumerate(rows):
        if not (start_date <= row.trade_date <= end_date):
            continue
        anchor_indices.append(index)
        future = rows[index + 1 : index + 1 + lookahead_days]
        if len(future) < lookahead_days:
            continue
        min_future_low = min(bar.low for bar in future)
        drop_rate = (min_future_low - row.close) / row.close
        if drop_rate <= -drop_threshold:
            positives.append(SampleEvent(row.scode, row.trade_date, 1, "drop6_positive", drop_rate))
            left = max(0, index - negative_exclude_days)
            right = min(len(rows) - 1, index + negative_exclude_days)
            excluded_negative_indices.update(range(left, right + 1))

    for index in anchor_indices:
        if index in excluded_negative_indices:
            continue
        row = rows[index]
        future = rows[index + 1 : index + 1 + lookahead_days]
        if len(future) < lookahead_days:
            continue
        min_future_low = min(bar.low for bar in future)
        drop_rate = (min_future_low - row.close) / row.close
        negative_candidates.append(SampleEvent(row.scode, row.trade_date, 0, "drop6_negative", drop_rate))

    return positives, negative_candidates


def load_drop6_events(
    conn,
    start_date: date,
    end_date: date,
    positive_limit: int | None,
    negative_ratio: float,
    seed: int,
) -> tuple[list[SampleEvent], list[SampleEvent]]:
    rows_by_symbol = _load_daily_bars_by_symbol(conn, start_date, end_date)
    positives: list[SampleEvent] = []
    negative_candidates: list[SampleEvent] = []
    for rows in rows_by_symbol.values():
        symbol_positives, symbol_negatives = _build_events_for_symbol(rows, start_date, end_date)
        positives.extend(symbol_positives)
        negative_candidates.extend(symbol_negatives)

    rng = random.Random(seed)
    positives.sort(key=lambda event: (event.anchor_date, event.scode))
    if positive_limit is not None:
        positives = sorted(positives, key=lambda event: (rng.random(), event.anchor_date, event.scode))[:positive_limit]
        positives.sort(key=lambda event: (event.anchor_date, event.scode))

    negative_limit = max(1, int(len(positives) * negative_ratio)) if positives else 0
    negatives = sorted(negative_candidates, key=lambda event: (rng.random(), event.anchor_date, event.scode))[:negative_limit]
    negatives.sort(key=lambda event: (event.anchor_date, event.scode))
    return positives, negatives


def split_train_test(samples: list[dict], train_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if train_ratio >= 0.999:
        return samples, []
    if train_ratio <= 0.001:
        return [], samples
    rng = random.Random(seed)
    rows = list(samples)
    rng.shuffle(rows)
    split_at = int(len(rows) * train_ratio)
    return rows[:split_at], rows[split_at:]


def build_drop6_dataset(
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
    del stat_type
    with mysql_connect() as conn:
        positives, negatives = load_drop6_events(conn, start_date, end_date, positive_limit, negative_ratio, seed)
        all_events = positives + negatives
        print(f"loaded events drop6_positives={len(positives)} negatives={len(negatives)}", flush=True)
        samples = materialize_events(
            conn,
            all_events,
            daily_window,
            weekly_window,
            batch_size,
            sample_mode=sample_mode,
            monthly_window=monthly_window,
        )
    train_rows, test_rows = split_train_test(samples, train_ratio, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_jsonl(output_dir / "all.jsonl", samples)
    print(
        f"dataset written dir={output_dir} train={len(train_rows)} test={len(test_rows)} all={len(samples)} "
        f"drop_threshold={DROP_THRESHOLD} negative_exclude_trading_days={NEGATIVE_EXCLUDE_TRADING_DAYS}",
        flush=True,
    )
    return len(train_rows), len(test_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build drop6 black-box fine-tuning training samples")
    parser.add_argument("--output-dir", type=Path, default=default_data_dir(os.environ.get("SAMPLE_MODE")))
    parser.add_argument("--stat-type", default=DEFAULT_STAT_TYPE)
    parser.add_argument("--start-date", default=os.environ.get("TRAIN_START_DATE", DEFAULT_TRAIN_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("TRAIN_END_DATE", DEFAULT_TRAIN_END_DATE))
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=float(os.environ.get("NEGATIVE_RATIO", "9.0")),
        help="Number of randomly selected non-drop samples relative to drop6 positive samples.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAIN_SEED", str(DEFAULT_TRAIN_SEED))))
    parser.add_argument("--sample-mode", choices=["short", "long", "xlong", "xxlong"], default=DEFAULT_SAMPLE_MODE)
    parser.add_argument("--daily-window", type=int, help="Override daily bars for the selected sample mode")
    parser.add_argument("--weekly-window", type=int, help="Override weekly bars for the selected sample mode")
    parser.add_argument("--monthly-window", type=int, help="Override monthly bars for the selected sample mode")
    parser.add_argument("--batch-size", type=int, default=80)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_drop6_dataset(
        output_dir=args.output_dir,
        stat_type=args.stat_type,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        positive_limit=args.positive_limit,
        negative_ratio=max(0.0, args.negative_ratio),
        train_ratio=min(max(args.train_ratio, 0.0), 1.0),
        seed=args.seed,
        daily_window=max(2, args.daily_window) if args.daily_window else None,
        weekly_window=max(2, args.weekly_window) if args.weekly_window else None,
        monthly_window=max(0, args.monthly_window) if args.monthly_window is not None else None,
        batch_size=max(1, args.batch_size),
        sample_mode=args.sample_mode,
    )


if __name__ == "__main__":
    main()
