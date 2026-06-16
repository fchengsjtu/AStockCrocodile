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

from blackbox_finetune.build_dataset import split_train_test, stable_rank
from blackbox_finetune_down6_neutral.common import (
    DEFAULT_SAMPLE_MODE,
    DEFAULT_TRAIN_END_DATE,
    DEFAULT_TRAIN_SEED,
    DEFAULT_TRAIN_START_DATE,
    SampleEvent,
    default_data_dir,
    materialize_events,
    mysql_connect,
    parse_date,
    write_jsonl,
)

POSITIVE_GAIN = 0.20
DOWN_DROP = 0.06
FUTURE_TRADING_DAYS = 3
POSITIVE_EXCLUSION_TRADING_DAYS = 20


@dataclass(frozen=True)
class ClassifiedCandidate:
    scode: str
    anchor_date: date
    symbol_index: int
    is_positive_surge: bool
    is_down6: bool


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


def _classify_candidate(row: tuple, symbol_index: int) -> ClassifiedCandidate | None:
    close = float(row[2] or 0)
    values = row[3:9]
    if close <= 0 or any(value is None for value in values):
        return None
    future_highs = [float(row[3]), float(row[5]), float(row[7])]
    future_lows = [float(row[4]), float(row[6]), float(row[8])]
    return ClassifiedCandidate(
        scode=str(row[0]),
        anchor_date=parse_date(row[1]),
        symbol_index=symbol_index,
        is_positive_surge=max(future_highs[:FUTURE_TRADING_DAYS]) >= close * (1.0 + POSITIVE_GAIN),
        is_down6=min(future_lows[:FUTURE_TRADING_DAYS]) <= close * (1.0 - DOWN_DROP),
    )


def iter_down6_neutral_events(
    conn,
    start_date: date,
    end_date: date,
    candidate_batch_size: int = 80,
    mysql_query_retries: int = 3,
) -> Iterator[SampleEvent]:
    query_start = start_date
    query_end = end_date + timedelta(days=20)
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
        by_symbol: dict[str, list[ClassifiedCandidate]] = {}
        symbol_indexes: dict[str, int] = {}
        for row in rows:
            scode = str(row[0])
            symbol_index = symbol_indexes.get(scode, -1) + 1
            symbol_indexes[scode] = symbol_index
            candidate = _classify_candidate(row, symbol_index)
            if candidate is not None:
                by_symbol.setdefault(scode, []).append(candidate)

        down_count = neutral_count = excluded_neutral = 0
        for scode, candidates in by_symbol.items():
            positive_indexes = [candidate.symbol_index for candidate in candidates if candidate.is_positive_surge]
            for candidate in candidates:
                if candidate.is_down6:
                    down_count += 1
                    yield SampleEvent(candidate.scode, candidate.anchor_date, 1, "down6", None)
                    continue
                if candidate.is_positive_surge:
                    excluded_neutral += 1
                    continue
                in_positive_window = any(
                    abs(candidate.symbol_index - positive_index) <= POSITIVE_EXCLUSION_TRADING_DAYS
                    for positive_index in positive_indexes
                )
                if in_positive_window:
                    excluded_neutral += 1
                    continue
                neutral_count += 1
                yield SampleEvent(candidate.scode, candidate.anchor_date, 0, "neutral", None)
        print(
            f"candidate batch={batch_index}/{len(batches)} symbols={len(batch)} rows={len(rows)} "
            f"down6={down_count} neutral={neutral_count} excluded_neutral={excluded_neutral}",
            flush=True,
        )


def _ranked_events(events: list[SampleEvent], seed: int, tag: str) -> list[SampleEvent]:
    return sorted(events, key=lambda event: stable_rank(seed, tag, event.scode, event.anchor_date))


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


def _balanced_rows(rows: list[dict], neutral_ratio: float, seed: int, down6_limit: int | None) -> list[dict]:
    down_rows = [row for row in rows if int(row["metadata"]["label"]) == 1]
    neutral_rows = [row for row in rows if int(row["metadata"]["label"]) == 0]
    down_count = len(down_rows)
    if down6_limit and down6_limit > 0:
        down_count = min(down_count, down6_limit)
    down_count = min(down_count, int(len(neutral_rows) / max(neutral_ratio, 0.001)) if neutral_ratio > 0 else down_count)
    if down_count <= 0:
        raise RuntimeError(f"Unable to build down6/neutral dataset: down6={len(down_rows)} neutral={len(neutral_rows)}")
    neutral_count = int(round(down_count * max(neutral_ratio, 0.0)))
    selected_down = sorted(
        down_rows,
        key=lambda row: stable_rank(seed, "down6", row["metadata"]["scode"], row["metadata"]["anchor_date"]),
    )[:down_count]
    selected_neutral = sorted(
        neutral_rows,
        key=lambda row: stable_rank(seed, "neutral", row["metadata"]["scode"], row["metadata"]["anchor_date"]),
    )[:neutral_count]
    selected = selected_down + selected_neutral
    selected.sort(key=lambda row: stable_rank(seed, "selected", row["metadata"]["scode"], row["metadata"]["anchor_date"]))
    return selected


def build_down6_neutral_dataset(
    output_dir: Path,
    start_date: date,
    end_date: date,
    down6_limit: int | None,
    neutral_ratio: float,
    train_ratio: float,
    seed: int,
    daily_window: int | None,
    weekly_window: int | None,
    monthly_window: int | None,
    batch_size: int,
    sample_mode: str,
    candidate_batch_size: int,
    mysql_query_retries: int,
) -> tuple[int, int]:
    with mysql_connect() as conn:
        down_events: list[SampleEvent] = []
        neutral_events: list[SampleEvent] = []
        for index, event in enumerate(
            iter_down6_neutral_events(conn, start_date, end_date, candidate_batch_size, mysql_query_retries),
            start=1,
        ):
            if event.label == 1:
                down_events.append(event)
            else:
                neutral_events.append(event)
            if index % 100000 == 0:
                print(f"classified candidates={index} down6={len(down_events)} neutral={len(neutral_events)}", flush=True)
        down_target = down6_limit if down6_limit and down6_limit > 0 else None
        down_rows = _materialize_until(
            conn,
            down_events,
            down_target,
            seed,
            "down6",
            daily_window,
            weekly_window,
            monthly_window,
            batch_size,
            sample_mode,
        )
        usable_down_count = len(down_rows)
        if down6_limit and down6_limit > 0:
            usable_down_count = min(usable_down_count, down6_limit)
        neutral_rows = _materialize_until(
            conn,
            neutral_events,
            int(round(usable_down_count * max(neutral_ratio, 0.0))),
            seed,
            "neutral",
            daily_window,
            weekly_window,
            monthly_window,
            batch_size,
            sample_mode,
        )
    samples = _balanced_rows(down_rows + neutral_rows, neutral_ratio, seed, down6_limit)
    train_rows, test_rows = split_train_test(samples, train_ratio, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_jsonl(output_dir / "all.jsonl", samples)
    down_count = sum(1 for row in samples if int(row["metadata"]["label"]) == 1)
    neutral_count = len(samples) - down_count
    print(
        f"dataset written dir={output_dir} train={len(train_rows)} test={len(test_rows)} "
        f"all={len(samples)} down6={down_count} neutral={neutral_count} neutral_ratio={neutral_ratio}",
        flush=True,
    )
    return len(train_rows), len(test_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build down6-vs-neutral black-box fine-tuning samples")
    parser.add_argument("--output-dir", type=Path, default=default_data_dir(os.environ.get("SAMPLE_MODE")))
    parser.add_argument("--start-date", default=os.environ.get("TRAIN_START_DATE", DEFAULT_TRAIN_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("TRAIN_END_DATE", DEFAULT_TRAIN_END_DATE))
    parser.add_argument("--down6-limit", "--positive-limit", dest="down6_limit", type=int)
    parser.add_argument(
        "--neutral-ratio",
        "--negative-ratio",
        dest="neutral_ratio",
        type=float,
        default=float(os.environ.get("NEUTRAL_RATIO", os.environ.get("NEGATIVE_RATIO", "9.0"))),
        help="Number of neutral samples relative to down6 samples.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAIN_SEED", str(DEFAULT_TRAIN_SEED))))
    parser.add_argument("--sample-mode", choices=["short", "long", "xlong", "xxlong"], default=DEFAULT_SAMPLE_MODE)
    parser.add_argument("--daily-window", type=int)
    parser.add_argument("--weekly-window", type=int)
    parser.add_argument("--monthly-window", type=int)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--candidate-batch-size", type=int, default=int(os.environ.get("CANDIDATE_BATCH_SIZE", "80")))
    parser.add_argument("--mysql-query-retries", type=int, default=int(os.environ.get("MYSQL_QUERY_RETRIES", "3")))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_down6_neutral_dataset(
        output_dir=args.output_dir,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        down6_limit=args.down6_limit,
        neutral_ratio=max(0.0, args.neutral_ratio),
        train_ratio=min(max(args.train_ratio, 0.01), 0.99),
        seed=args.seed,
        daily_window=max(2, args.daily_window) if args.daily_window else None,
        weekly_window=max(2, args.weekly_window) if args.weekly_window else None,
        monthly_window=max(0, args.monthly_window) if args.monthly_window is not None else None,
        batch_size=max(1, args.batch_size),
        sample_mode=args.sample_mode,
        candidate_batch_size=max(1, args.candidate_batch_size),
        mysql_query_retries=max(1, args.mysql_query_retries),
    )


if __name__ == "__main__":
    main()
