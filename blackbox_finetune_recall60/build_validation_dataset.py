from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_recall60.build_dataset import build_recall60_dataset
from blackbox_finetune_recall60.common import (
    DEFAULT_STAT_TYPE,
    DEFAULT_VALIDATION_DIR,
    DEFAULT_VALIDATION_END_DATE,
    DEFAULT_VALIDATION_START_DATE,
    DEFAULT_SAMPLE_MODE,
    parse_date,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build recall60 validation dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--stat-type", default=DEFAULT_STAT_TYPE)
    parser.add_argument("--start-date", default=os.environ.get("VALIDATION_START_DATE") or os.environ.get("TEST_START_DATE", DEFAULT_VALIDATION_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("VALIDATION_END_DATE") or os.environ.get("TEST_END_DATE", DEFAULT_VALIDATION_END_DATE))
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--negative-ratio", type=float, default=float(os.environ.get("NEGATIVE_RATIO", "3.0")))
    parser.add_argument("--sample-mode", choices=["short", "long", "xlong"], default=DEFAULT_SAMPLE_MODE)
    parser.add_argument("--daily-window", type=int, help="Override daily bars for the selected sample mode")
    parser.add_argument("--weekly-window", type=int, help="Override weekly bars for the selected sample mode")
    parser.add_argument("--monthly-window", type=int, help="Override monthly bars for the selected sample mode")
    parser.add_argument("--batch-size", type=int, default=80)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_recall60_dataset(
        output_dir=args.output_dir,
        stat_type=args.stat_type,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        positive_limit=args.positive_limit,
        negative_ratio=max(0.0, args.negative_ratio),
        train_ratio=0.01,
        seed=20260518,
        daily_window=max(2, args.daily_window) if args.daily_window else None,
        weekly_window=max(2, args.weekly_window) if args.weekly_window else None,
        monthly_window=max(0, args.monthly_window) if args.monthly_window is not None else None,
        batch_size=max(1, args.batch_size),
        sample_mode=args.sample_mode,
    )


if __name__ == "__main__":
    main()
