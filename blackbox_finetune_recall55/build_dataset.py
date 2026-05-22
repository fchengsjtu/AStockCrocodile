from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune.build_dataset import (
    load_negative_events,
    load_positive_events,
    split_train_test,
)
from blackbox_finetune_recall55.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_STAT_TYPE,
    DEFAULT_TRAIN_END_DATE,
    DEFAULT_TRAIN_START_DATE,
    DEFAULT_WINDOW,
    materialize_events,
    mysql_connect,
    parse_date,
    write_jsonl,
)

DEFAULT_SEED = 20260518


def build_recall55_dataset(
    output_dir: Path,
    stat_type,
    start_date,
    end_date,
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
    parser = argparse.ArgumentParser(description="Build recall55 black-box fine-tuning training samples")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stat-type", default=DEFAULT_STAT_TYPE)
    parser.add_argument("--start-date", default=DEFAULT_TRAIN_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_TRAIN_END_DATE)
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
    build_recall55_dataset(
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
