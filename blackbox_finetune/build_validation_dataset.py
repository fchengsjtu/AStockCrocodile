from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune.build_dataset import build_dataset
from blackbox_finetune.common import DEFAULT_STAT_TYPE, DEFAULT_WINDOW, parse_date


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build validation dataset for 2026 holdout")
    parser.add_argument("--output-dir", type=Path, default=Path("blackbox_finetune") / "data_validation")
    parser.add_argument("--stat-type", default=DEFAULT_STAT_TYPE)
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20260430")
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--daily-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--batch-size", type=int, default=80)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_dataset(
        output_dir=args.output_dir,
        stat_type=args.stat_type,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        positive_limit=args.positive_limit,
        negative_ratio=max(0.0, args.negative_ratio),
        train_ratio=0.01,
        seed=20260517,
        daily_window=max(2, args.daily_window),
        weekly_window=max(2, args.weekly_window),
        batch_size=max(1, args.batch_size),
    )
    # build_dataset writes almost all rows to test.jsonl when train_ratio=0.01.


if __name__ == "__main__":
    main()

