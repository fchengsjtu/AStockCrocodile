from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_down6_neutral.build_dataset import build_down6_neutral_dataset
from blackbox_finetune_down6_neutral.common import (
    DEFAULT_SAMPLE_MODE,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VALIDATION_END_DATE,
    DEFAULT_VALIDATION_START_DATE,
    default_validation_dir,
    parse_date,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build down6-vs-neutral validation dataset")
    parser.add_argument("--output-dir", type=Path, default=default_validation_dir(os.environ.get("SAMPLE_MODE")))
    parser.add_argument("--start-date", default=os.environ.get("VALIDATION_START_DATE") or os.environ.get("TEST_START_DATE", DEFAULT_VALIDATION_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("VALIDATION_END_DATE") or os.environ.get("TEST_END_DATE", DEFAULT_VALIDATION_END_DATE))
    parser.add_argument("--down6-limit", "--positive-limit", dest="down6_limit", type=int)
    parser.add_argument("--neutral-ratio", "--negative-ratio", dest="neutral_ratio", type=float, default=float(os.environ.get("NEUTRAL_RATIO", os.environ.get("NEGATIVE_RATIO", "9.0"))))
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
        train_ratio=0.01,
        seed=int(os.environ.get("TRAIN_SEED", str(DEFAULT_TRAIN_SEED))),
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
