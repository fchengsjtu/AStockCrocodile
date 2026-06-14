from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from blackbox_finetune_threeclass.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VALIDATION_DIR,
    compact_messages_from_sample,
    read_jsonl,
)
from blackbox_finetune_threeclass.gpu import prepare_rtx3060
from blackbox_finetune_threeclass.inference import load_model, score_prediction
from blackbox_finetune_threeclass.metrics import summarize_scored_rows


def evaluate_dataset(
    base_model: str,
    adapter_dir: Path,
    data_dir: Path,
    max_samples: int | None,
    max_seq_length: int,
    output: Path | None,
) -> dict:
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"missing test dataset: {test_path}")
    rows = read_jsonl(test_path, max_samples)
    model, tokenizer = load_model(base_model, adapter_dir)
    scored = []
    for index, row in enumerate(rows, start=1):
        messages = compact_messages_from_sample(row)
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prediction = score_prediction(model, tokenizer, prompt, max_seq_length)
        scored.append(
            {
                "scode": row["metadata"].get("scode"),
                "anchor_date": row["metadata"].get("anchor_date"),
                "actual_label": int(row["metadata"]["label"]),
                "predicted_label": int(prediction["label_id"]),
                "positive_probability": prediction["positive_probability"],
                "negative_probability": prediction["negative_probability"],
                "neutral_probability": prediction["neutral_probability"],
            }
        )
        if index % 100 == 0 or index == len(rows):
            running = summarize_scored_rows(scored)
            print(
                f"evaluation progress {index}/{len(rows)} accuracy={running['accuracy']:.4f} "
                f"macro_f1={running['macro_f1']:.4f} "
                f"positive_precision@10={running['positive_precision@10']:.4f}",
                flush=True,
            )
    result = summarize_scored_rows(scored)
    result["max_seq_length"] = max_seq_length
    result_path = output or (adapter_dir.parent / "evaluations" / "threeclass-evaluation.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result["output_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the three-class black-box model")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "adapter")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--allow-non-rtx3060", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    evaluate_dataset(
        args.base_model,
        args.adapter_dir,
        args.data_dir,
        args.max_samples,
        max(64, args.max_seq_length),
        args.output,
    )


if __name__ == "__main__":
    main()

