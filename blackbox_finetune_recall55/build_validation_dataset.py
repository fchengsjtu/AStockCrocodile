from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_recall55.build_dataset import build_recall55_dataset
from blackbox_finetune_recall55.common import (
    DEFAULT_STAT_TYPE,
    DEFAULT_VALIDATION_DIR,
    DEFAULT_VALIDATION_END_DATE,
    DEFAULT_VALIDATION_START_DATE,
    DEFAULT_WINDOW,
    parse_date,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build recall55 validation dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--stat-type", default=DEFAULT_STAT_TYPE)
    parser.add_argument("--start-date", default=DEFAULT_VALIDATION_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_VALIDATION_END_DATE)
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
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
        train_ratio=0.01,
        seed=20260518,
        daily_window=max(2, args.daily_window),
        weekly_window=max(2, args.weekly_window),
        batch_size=max(1, args.batch_size),
    )


if __name__ == "__main__":
    main()
