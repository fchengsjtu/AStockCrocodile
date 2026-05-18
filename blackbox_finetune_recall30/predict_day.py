from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune.predict_day import predict_day
from blackbox_finetune_recall30.common import DEFAULT_BASE_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_WINDOW


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one trading day with recall30 black-box model")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "adapter")
    parser.add_argument("--date", dest="trade_date", required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--daily-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    predict_day(
        args.base_model,
        args.adapter_dir,
        args.trade_date,
        args.threshold,
        max(2, args.daily_window),
        max(2, args.weekly_window),
        max(1, args.batch_size),
        args.limit,
        args.output,
    )


if __name__ == "__main__":
    main()
