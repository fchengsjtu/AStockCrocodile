from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_recall50.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MIN_POSITIVE_RECALL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VALIDATION_DIR,
    read_jsonl,
)
from blackbox_finetune_recall50.gpu import prepare_rtx3060
from blackbox_finetune_recall50.inference import load_model, score_prediction


def evaluate_dataset(
    base_model: str,
    adapter_dir: Path,
    data_dir: Path,
    threshold: float,
    min_positive_recall: float,
    max_samples: int | None,
    max_seq_length: int,
) -> dict:
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"missing test dataset: {test_path}")
    rows = read_jsonl(test_path, max_samples)
    model, tokenizer = load_model(base_model, adapter_dir)
    tp = fp = tn = fn = positives = 0
    for idx, row in enumerate(rows, start=1):
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        pred = score_prediction(model, tokenizer, prompt, max_seq_length, threshold)
        predicted_positive = pred["label"] == "positive"
        actual_positive = int(row["metadata"]["label"]) == 1
        if actual_positive:
            positives += 1
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1
        print(f"eval {idx}/{len(rows)} actual={int(actual_positive)} pred={pred['label']} p={pred['positive_probability']:.4f}", flush=True)
    positive_recall = tp / positives if positives else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    result = {
        "samples": len(rows),
        "positive_samples": positives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_recall": positive_recall,
        "precision": precision,
        "threshold": threshold,
        "min_positive_recall": min_positive_recall,
        "max_seq_length": max_seq_length,
        "passed": positive_recall >= min_positive_recall,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(f"positive_recall={positive_recall:.4f} < {min_positive_recall:.4f}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate recall50 black-box fine-tuned classifier")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "adapter")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-positive-recall", type=float, default=DEFAULT_MIN_POSITIVE_RECALL)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--cuda-device", default="0", help="CUDA device id. Default binds the RTX3060 as cuda:0.")
    parser.add_argument("--allow-non-rtx3060", action="store_true", help="Allow CUDA devices whose name is not RTX 3060.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    evaluate_dataset(
        args.base_model,
        args.adapter_dir,
        args.data_dir,
        args.threshold,
        args.min_positive_recall,
        args.max_samples,
        max(64, args.max_seq_length),
    )


if __name__ == "__main__":
    main()
