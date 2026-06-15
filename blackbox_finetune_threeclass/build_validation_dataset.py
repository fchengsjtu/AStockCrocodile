from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from blackbox_finetune_threeclass.build_dataset import build_threeclass_dataset
from blackbox_finetune_threeclass.common import (
    DEFAULT_SAMPLE_MODE,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VALIDATION_DIR,
    DEFAULT_VALIDATION_END_DATE,
    DEFAULT_VALIDATION_START_DATE,
    parse_date,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the three-class holdout dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--start-date", default=os.environ.get("VALIDATION_START_DATE", DEFAULT_VALIDATION_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("VALIDATION_END_DATE", DEFAULT_VALIDATION_END_DATE))
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("EVAL_RANDOM_SEED", str(DEFAULT_TRAIN_SEED))))
    parser.add_argument("--sample-mode", choices=["short", "long", "xlong", "xxlong"], default=os.environ.get("SAMPLE_MODE", DEFAULT_SAMPLE_MODE))
    parser.add_argument("--daily-window", type=int)
    parser.add_argument("--weekly-window", type=int)
    parser.add_argument("--monthly-window", type=int)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--candidate-batch-size", type=int, default=int(os.environ.get("CANDIDATE_BATCH_SIZE", "80")))
    parser.add_argument("--mysql-query-retries", type=int, default=int(os.environ.get("MYSQL_QUERY_RETRIES", "3")))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_threeclass_dataset(
        output_dir=args.output_dir,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        positive_limit=args.positive_limit,
        train_ratio=0.01,
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
