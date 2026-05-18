from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune.evaluate import evaluate_dataset
from blackbox_finetune_recall30.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MIN_POSITIVE_RECALL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VALIDATION_DIR,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate recall30 black-box fine-tuned classifier")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "adapter")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-positive-recall", type=float, default=DEFAULT_MIN_POSITIVE_RECALL)
    parser.add_argument("--max-samples", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate_dataset(
        args.base_model,
        args.adapter_dir,
        args.data_dir,
        args.threshold,
        args.min_positive_recall,
        args.max_samples,
    )


if __name__ == "__main__":
    main()
