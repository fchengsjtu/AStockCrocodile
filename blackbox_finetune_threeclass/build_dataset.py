from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune.build_dataset import stable_rank
from blackbox_finetune_threeclass.common import (
    CLASS_NAMES,
    CLASS_NEGATIVE,
    CLASS_NEUTRAL,
    CLASS_POSITIVE,
    DEFAULT_DATA_DIR,
    DEFAULT_SAMPLE_MODE,
    DEFAULT_TRAIN_END_DATE,
    DEFAULT_TRAIN_SEED,
    DEFAULT_TRAIN_START_DATE,
    SampleEvent,
    materialize_events,
    mysql_connect,
    parse_date,
    write_jsonl,
)

POSITIVE_GAIN = 0.20
NEGATIVE_DROP = 0.06
FUTURE_TRADING_DAYS = 3
POSITIVE_COOLDOWN_TRADING_DAYS = 20


@dataclass(frozen=True)
class FutureBar:
    high: float
    low: float


def classify_future_path(
    anchor_close: float,
    future_bars: list[FutureBar],
    positive_gain: float = POSITIVE_GAIN,
    negative_drop: float = NEGATIVE_DROP,
) -> int | None:
    if anchor_close <= 0 or len(future_bars) < FUTURE_TRADING_DAYS:
        return None
    positive_price = anchor_close * (1.0 + positive_gain)
    negative_price = anchor_close * (1.0 - negative_drop)
    for bar in future_bars[:FUTURE_TRADING_DAYS]:
        positive_hit = bar.high >= positive_price
        negative_hit = bar.low <= negative_price
        if positive_hit and negative_hit:
            return None
        if positive_hit:
            return CLASS_POSITIVE
        if negative_hit:
            return CLASS_NEGATIVE
    return CLASS_NEUTRAL


def _candidate_sql(symbol_count: int) -> str:
    placeholders = ",".join(["%s"] * symbol_count)
    return f"""
        SELECT SCode, TradeDate, ClosePrice,
               High1, Low1, High2, Low2, High3, Low3
        FROM (
            SELECT SCode,
                   DATE(KTime) AS TradeDate,
                   Close AS ClosePrice,
                   LEAD(High, 1) OVER (PARTITION BY SCode ORDER BY KTime) AS High1,
                   LEAD(Low, 1) OVER (PARTITION BY SCode ORDER BY KTime) AS Low1,
                   LEAD(High, 2) OVER (PARTITION BY SCode ORDER BY KTime) AS High2,
                   LEAD(Low, 2) OVER (PARTITION BY SCode ORDER BY KTime) AS Low2,
                   LEAD(High, 3) OVER (PARTITION BY SCode ORDER BY KTime) AS High3,
                   LEAD(Low, 3) OVER (PARTITION BY SCode ORDER BY KTime) AS Low3
            FROM dkandles
            WHERE KType = 'D'
              AND SCode IN ({placeholders})
              AND KTime >= %s
              AND KTime < %s
        ) ranked
        WHERE TradeDate >= %s AND TradeDate <= %s
        ORDER BY SCode, TradeDate
    """


def load_candidate_symbols(conn, start_date: date, end_date: date) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT SCode
            FROM dkandles
            WHERE KType = 'D'
              AND KTime >= %s
              AND KTime < %s
            ORDER BY SCode
            """,
            (start_date, end_date + timedelta(days=21)),
        )
        return [str(row[0]) for row in cur.fetchall()]


def _load_candidate_rows(
    conn,
    symbols: list[str],
    query_start: date,
    query_end: date,
    start_date: date,
    end_date: date,
    max_retries: int,
) -> list[tuple]:
    params = [*symbols, query_start, query_end, start_date, end_date]
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            with conn.cursor() as cur:
                cur.execute(_candidate_sql(len(symbols)), params)
                return list(cur.fetchall())
        except Exception as exc:
            error_code = exc.args[0] if getattr(exc, "args", None) else None
            if error_code not in {2006, 2013, 2055} or attempt >= max(1, max_retries):
                raise
            delay = min(5, attempt)
            print(
                f"candidate query lost MySQL connection attempt={attempt}/{max_retries} "
                f"symbols={len(symbols)} error={exc}; reconnecting in {delay}s",
                flush=True,
            )
            time.sleep(delay)
            conn.ping(reconnect=True)
    return []


def iter_classified_events(
    conn,
    start_date: date,
    end_date: date,
    candidate_batch_size: int = 80,
    mysql_query_retries: int = 3,
) -> Iterator[SampleEvent]:
    query_start = start_date
    query_end = end_date + timedelta(days=20)
    last_positive_index: dict[str, int] = {}
    symbol_indexes: dict[str, int] = {}
    symbols = load_candidate_symbols(conn, start_date, end_date)
    batches = [
        symbols[start : start + max(1, candidate_batch_size)]
        for start in range(0, len(symbols), max(1, candidate_batch_size))
    ]
    print(
        f"candidate scan symbols={len(symbols)} batches={len(batches)} "
        f"batch_size={max(1, candidate_batch_size)} retries={max(1, mysql_query_retries)}",
        flush=True,
    )
    for batch_index, batch in enumerate(batches, start=1):
        rows = _load_candidate_rows(
            conn,
            batch,
            query_start,
            query_end,
            start_date,
            end_date,
            mysql_query_retries,
        )
        batch_events = 0
        for row in rows:
            scode = str(row[0])
            symbol_index = symbol_indexes.get(scode, -1) + 1
            symbol_indexes[scode] = symbol_index
            values = row[3:9]
            if any(value is None for value in values):
                continue
            future = [
                FutureBar(float(row[3]), float(row[4])),
                FutureBar(float(row[5]), float(row[6])),
                FutureBar(float(row[7]), float(row[8])),
            ]
            label = classify_future_path(float(row[2] or 0), future)
            if label is None:
                continue
            if label == CLASS_POSITIVE:
                previous = last_positive_index.get(scode)
                if previous is not None and symbol_index - previous <= POSITIVE_COOLDOWN_TRADING_DAYS:
                    continue
                last_positive_index[scode] = symbol_index
            batch_events += 1
            yield SampleEvent(
                scode=scode,
                anchor_date=parse_date(row[1]),
                label=label,
                source=CLASS_NAMES[label],
                gain_rate=None,
            )
        print(
            f"candidate batch={batch_index}/{len(batches)} symbols={len(batch)} "
            f"rows={len(rows)} classified={batch_events}",
            flush=True,
        )


def _ranked_events(events: list[SampleEvent], seed: int, tag: str) -> list[SampleEvent]:
    return sorted(
        events,
        key=lambda event: stable_rank(seed, tag, event.scode, event.anchor_date),
    )


def _materialize_until(
    conn,
    events: list[SampleEvent],
    target: int | None,
    seed: int,
    tag: str,
    daily_window: int | None,
    weekly_window: int | None,
    monthly_window: int | None,
    batch_size: int,
    sample_mode: str,
) -> list[dict]:
    ordered = _ranked_events(events, seed, tag)
    if target is None:
        chunks = [ordered]
    else:
        chunk_size = min(20000, max(500, target * 2))
        chunks = [ordered[start : start + chunk_size] for start in range(0, len(ordered), chunk_size)]
    materialized: list[dict] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        if not chunk:
            continue
        materialized.extend(
            materialize_events(
                conn,
                chunk,
                daily_window,
                weekly_window,
                batch_size,
                sample_mode=sample_mode,
                monthly_window=monthly_window,
            )
        )
        print(
            f"materialized class={tag} chunk={chunk_index}/{len(chunks)} "
            f"usable={len(materialized)} target={target if target is not None else 'all'}",
            flush=True,
        )
        if target is not None and len(materialized) >= target:
            break
    return materialized


def _stratified_split(rows: list[dict], train_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for label in (CLASS_POSITIVE, CLASS_NEGATIVE, CLASS_NEUTRAL):
        class_rows = [row for row in rows if int(row["metadata"]["label"]) == label]
        class_rows.sort(
            key=lambda row: stable_rank(
                seed,
                label,
                row["metadata"]["scode"],
                row["metadata"]["anchor_date"],
            )
        )
        test_count = max(1, int(round(len(class_rows) * (1.0 - train_ratio)))) if len(class_rows) > 1 else 0
        test_rows.extend(class_rows[:test_count])
        train_rows.extend(class_rows[test_count:])
    train_rows = interleave_class_rows(train_rows, seed, "train")
    test_rows.sort(key=lambda row: stable_rank(seed, "test", row["metadata"]["scode"], row["metadata"]["anchor_date"]))
    return train_rows, test_rows


def interleave_class_rows(rows: list[dict], seed: int, tag: str) -> list[dict]:
    grouped = {
        label: [row for row in rows if int(row["metadata"]["label"]) == label]
        for label in (CLASS_POSITIVE, CLASS_NEGATIVE, CLASS_NEUTRAL)
    }
    for label, class_rows in grouped.items():
        class_rows.sort(
            key=lambda row: stable_rank(
                seed,
                f"{tag}-{CLASS_NAMES[label]}",
                row["metadata"]["scode"],
                row["metadata"]["anchor_date"],
            )
        )
    pattern = [CLASS_POSITIVE] + [CLASS_NEGATIVE] * 4 + [CLASS_NEUTRAL] * 10
    positions = {label: 0 for label in grouped}
    ordered: list[dict] = []
    while any(positions[label] < len(grouped[label]) for label in grouped):
        made_progress = False
        for label in pattern:
            position = positions[label]
            if position < len(grouped[label]):
                ordered.append(grouped[label][position])
                positions[label] = position + 1
                made_progress = True
        if not made_progress:
            break
    return ordered


def rebalance_materialized_samples(rows: list[dict], seed: int, positive_limit: int | None = None) -> list[dict]:
    grouped = {
        label: [row for row in rows if int(row["metadata"]["label"]) == label]
        for label in (CLASS_POSITIVE, CLASS_NEGATIVE, CLASS_NEUTRAL)
    }
    positive_count = min(
        len(grouped[CLASS_POSITIVE]),
        len(grouped[CLASS_NEGATIVE]) // 4,
        len(grouped[CLASS_NEUTRAL]) // 10,
    )
    if positive_limit and positive_limit > 0:
        positive_count = min(positive_count, positive_limit)
    if positive_count <= 0:
        raise RuntimeError(
            "Unable to create a 1:4:10 dataset after K-line filtering: "
            f"positive={len(grouped[CLASS_POSITIVE])} "
            f"negative={len(grouped[CLASS_NEGATIVE])} neutral={len(grouped[CLASS_NEUTRAL])}"
        )
    selected: list[dict] = []
    for label, count in (
        (CLASS_POSITIVE, positive_count),
        (CLASS_NEGATIVE, positive_count * 4),
        (CLASS_NEUTRAL, positive_count * 10),
    ):
        ordered = sorted(
            grouped[label],
            key=lambda row: stable_rank(seed, "materialized", label, row["metadata"]["scode"], row["metadata"]["anchor_date"]),
        )
        selected.extend(ordered[:count])
    return interleave_class_rows(selected, seed, "materialized")


def build_threeclass_dataset(
    output_dir: Path,
    start_date: date,
    end_date: date,
    positive_limit: int | None,
    train_ratio: float,
    seed: int,
    daily_window: int | None,
    weekly_window: int | None,
    monthly_window: int | None,
    batch_size: int,
    sample_mode: str,
    candidate_batch_size: int,
    mysql_query_retries: int,
    output_split: str = "split",
) -> tuple[int, int]:
    with mysql_connect() as conn:
        grouped: dict[int, list[SampleEvent]] = {
            CLASS_POSITIVE: [],
            CLASS_NEGATIVE: [],
            CLASS_NEUTRAL: [],
        }
        for index, event in enumerate(
            iter_classified_events(
                conn,
                start_date,
                end_date,
                candidate_batch_size,
                mysql_query_retries,
            ),
            start=1,
        ):
            grouped[event.label].append(event)
            if index % 100000 == 0:
                print(
                    f"classified candidates={index} positive={len(grouped[CLASS_POSITIVE])} "
                    f"negative={len(grouped[CLASS_NEGATIVE])} neutral={len(grouped[CLASS_NEUTRAL])}",
                    flush=True,
                )
        positive_target = positive_limit if positive_limit and positive_limit > 0 else None
        positive_rows = _materialize_until(
            conn,
            grouped[CLASS_POSITIVE],
            positive_target,
            seed,
            CLASS_NAMES[CLASS_POSITIVE],
            daily_window,
            weekly_window,
            monthly_window,
            batch_size,
            sample_mode,
        )
        usable_positive_count = len(positive_rows)
        if positive_limit and positive_limit > 0:
            usable_positive_count = min(usable_positive_count, positive_limit)
        negative_rows = _materialize_until(
            conn,
            grouped[CLASS_NEGATIVE],
            usable_positive_count * 4,
            seed,
            CLASS_NAMES[CLASS_NEGATIVE],
            daily_window,
            weekly_window,
            monthly_window,
            batch_size,
            sample_mode,
        )
        neutral_rows = _materialize_until(
            conn,
            grouped[CLASS_NEUTRAL],
            usable_positive_count * 10,
            seed,
            CLASS_NAMES[CLASS_NEUTRAL],
            daily_window,
            weekly_window,
            monthly_window,
            batch_size,
            sample_mode,
        )
        materialized = positive_rows + negative_rows + neutral_rows
    samples = rebalance_materialized_samples(materialized, seed, positive_limit)
    if output_split == "train":
        train_rows = list(samples)
        test_rows: list[dict] = []
    elif output_split == "test":
        train_rows = []
        test_rows = list(samples)
    else:
        train_rows, test_rows = _stratified_split(samples, train_ratio, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_jsonl(output_dir / "all.jsonl", samples)
    counts = {
        CLASS_NAMES[label]: sum(1 for row in samples if int(row["metadata"]["label"]) == label)
        for label in (CLASS_POSITIVE, CLASS_NEGATIVE, CLASS_NEUTRAL)
    }
    print(
        f"dataset written dir={output_dir} train={len(train_rows)} test={len(test_rows)} "
        f"all={len(samples)} split={output_split} classes={counts}",
        flush=True,
    )
    return len(train_rows), len(test_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build positive/negative/neutral black-box samples at a strict 1:4:10 ratio")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--start-date", default=os.environ.get("TRAIN_START_DATE", DEFAULT_TRAIN_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("TRAIN_END_DATE", DEFAULT_TRAIN_END_DATE))
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--output-split", choices=["split", "train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAIN_SEED", str(DEFAULT_TRAIN_SEED))))
    parser.add_argument("--sample-mode", choices=["short", "long", "xlong", "xxlong"], default=os.environ.get("SAMPLE_MODE", DEFAULT_SAMPLE_MODE))
    parser.add_argument("--daily-window", type=int)
    parser.add_argument("--weekly-window", type=int)
    parser.add_argument("--monthly-window", type=int)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=int(os.environ.get("CANDIDATE_BATCH_SIZE", "80")),
        help="Number of symbols per future-path SQL query.",
    )
    parser.add_argument(
        "--mysql-query-retries",
        type=int,
        default=int(os.environ.get("MYSQL_QUERY_RETRIES", "3")),
        help="Reconnect and retry a candidate SQL batch after MySQL connection-loss errors.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_threeclass_dataset(
        output_dir=args.output_dir,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        positive_limit=args.positive_limit,
        train_ratio=min(max(args.train_ratio, 0.01), 0.99),
        seed=args.seed,
        daily_window=max(2, args.daily_window) if args.daily_window else None,
        weekly_window=max(2, args.weekly_window) if args.weekly_window else None,
        monthly_window=max(0, args.monthly_window) if args.monthly_window is not None else None,
        batch_size=max(1, args.batch_size),
        sample_mode=args.sample_mode,
        candidate_batch_size=max(1, args.candidate_batch_size),
        mysql_query_retries=max(1, args.mysql_query_retries),
        output_split=args.output_split,
    )


if __name__ == "__main__":
    main()
