from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_down6_neutral.common import (
    DEFAULT_BASE_MODEL,
    compact_messages_from_sample,
    read_jsonl,
    DEFAULT_MAX_SEQ_LENGTH,
    normalize_precision_threshold,
    normalize_precision_top_k,
    precision_at_k,
    precision_target_tag,
    REPORTED_PRECISION_KS,
    default_max_seq_length,
    default_output_dir,
    default_validation_dir,
)
from blackbox_finetune_down6_neutral.gpu import prepare_rtx3060
from blackbox_finetune_down6_neutral.inference import load_model, score_prediction

EVALUATION_CHUNK_SIZE = 1000


def summarize_scored_rows(scored_rows: list[dict], precision_top_k: int, down6_score_floor: float = 0.45) -> dict:
    tp = fp = tn = fn = positives = 0
    floor = min(max(float(down6_score_floor), 0.0), 1.0)
    down6_low_score_count = 0
    for row in scored_rows:
        actual_positive = int(row["actual_label"]) == 1
        predicted_positive = int(row["predicted_label"]) == 1
        if actual_positive:
            positives += 1
            if float(row.get("positive_probability", 0.0)) < floor:
                down6_low_score_count += 1
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    positive_recall = tp / positives if positives else 0.0
    down6_low_score_rate = down6_low_score_count / positives if positives else 0.0
    ks = tuple(dict.fromkeys((*REPORTED_PRECISION_KS, precision_top_k)))
    return {
        "samples": len(scored_rows),
        "positive_samples": positives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_recall": positive_recall,
        "precision": precision,
        "down6_score_floor": floor,
        "down6_low_score_count": down6_low_score_count,
        "down6_low_score_rate": down6_low_score_rate,
        **precision_at_k(scored_rows, ks),
    }


def evaluate_dataset(
    base_model: str,
    adapter_dir: Path,
    data_dir: Path,
    threshold: float,
    precision_top_k: int,
    precision_threshold: float,
    max_samples: int | None,
    max_seq_length: int,
    output_dir: Path | None = None,
    min_positive_recall: float | None = None,
    down6_score_floor: float = 0.45,
) -> dict:
    precision_top_k = normalize_precision_top_k(precision_top_k)
    precision_threshold = normalize_precision_threshold(precision_threshold)
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"missing test dataset: {test_path}")
    rows = read_jsonl(test_path, max_samples)
    model, tokenizer = load_model(base_model, adapter_dir)
    scored_rows: list[dict] = []
    chunks: list[dict] = []
    chunk_start = 0
    for idx, row in enumerate(rows, start=1):
        messages = compact_messages_from_sample(row)
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        pred = score_prediction(model, tokenizer, prompt, max_seq_length, threshold)
        predicted_positive = pred["label"] == "positive"
        actual_positive = int(row["metadata"]["label"]) == 1
        scored_rows.append(
            {
                "scode": row.get("metadata", {}).get("scode"),
                "anchor_date": row.get("metadata", {}).get("anchor_date"),
                "actual_label": 1 if actual_positive else 0,
                "predicted_label": 1 if predicted_positive else 0,
                "positive_probability": pred["positive_probability"],
            }
        )
        print(f"eval {idx}/{len(rows)} actual={int(actual_positive)} pred={pred['label']} p={pred['positive_probability']:.4f}", flush=True)
        if idx % EVALUATION_CHUNK_SIZE == 0 or idx == len(rows):
            chunk_rows = scored_rows[chunk_start:idx]
            chunk = summarize_scored_rows(chunk_rows, precision_top_k, down6_score_floor)
            chunk.update(
                {
                    "chunk_index": len(chunks) + 1,
                    "start_index": chunk_start + 1,
                    "end_index": idx,
                }
            )
            chunks.append(chunk)
            print(
                "eval chunk "
                f"{chunk['chunk_index']} rows={chunk['start_index']}-{chunk['end_index']} "
                f"samples={chunk['samples']} positives={chunk['positive_samples']} "
                f"tp={chunk['tp']} fp={chunk['fp']} tn={chunk['tn']} fn={chunk['fn']} "
                f"positive_recall={chunk['positive_recall']:.4f} precision={chunk['precision']:.4f} "
                f"down6_low_score_rate={chunk['down6_low_score_rate']:.4f} "
                f"precision@5={chunk['precision@5']:.4f} precision@10={chunk['precision@10']:.4f} "
                f"precision@20={chunk['precision@20']:.4f} precision@50={chunk['precision@50']:.4f}",
                flush=True,
            )
            chunk_start = idx
    summary = summarize_scored_rows(scored_rows, precision_top_k, down6_score_floor)
    target_key = f"precision@{precision_top_k}"
    result = {
        **summary,
        "chunks": chunks,
        "chunk_size": EVALUATION_CHUNK_SIZE,
        "threshold": threshold,
        "min_positive_recall": min_positive_recall,
        "precision_top_k": precision_top_k,
        "precision_threshold": precision_threshold,
        "max_seq_length": max_seq_length,
        "passed": summary[target_key] >= precision_threshold,
    }
    result_dir = output_dir or (adapter_dir.parent / "evaluations" / precision_target_tag(precision_top_k, precision_threshold))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "evaluation.json"
    result["output_path"] = str(result_path)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(f"{target_key}={summary[target_key]:.4f} < {precision_threshold:.4f}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate down6_neutral black-box fine-tuned classifier")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=default_output_dir() / "adapter")
    parser.add_argument("--data-dir", type=Path, default=default_validation_dir())
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--precision-top-k", type=int, default=normalize_precision_top_k(), help="Top-k bucket used for the precision gate. Default reads PRECISION_TOP_K, then 20.")
    parser.add_argument("--precision-threshold", type=float, default=normalize_precision_threshold(), help="Required precision for --precision-top-k. Default reads PRECISION_THRESHOLD, then MIN_PRECISION_AT_20/PRECISION_AT_20_TARGET, then 0.30.")
    parser.add_argument("--min-precision-at-20", type=float, default=None, help="Deprecated compatibility option. Equivalent to --precision-top-k 20 plus this precision threshold.")
    parser.add_argument("--min-positive-recall", type=float, default=None, help="Deprecated compatibility option. Reported in JSON when provided but no longer used as the pass/fail gate.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Override token length; default follows sample mode")
    parser.add_argument("--output-dir", type=Path, help="Directory to write evaluation.json. Default is adapter parent/evaluations/<precision target tag>.")
    parser.add_argument("--down6-score-floor", type=float, default=0.45, help="Report the share of true down6 samples below this probability floor.")
    parser.add_argument("--cuda-device", default="0", help="CUDA device id. Default binds the RTX3060 as cuda:0.")
    parser.add_argument("--allow-non-rtx3060", action="store_true", help="Allow CUDA devices whose name is not RTX 3060.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    precision_top_k = 20 if args.min_precision_at_20 is not None else normalize_precision_top_k(args.precision_top_k)
    precision_threshold = normalize_precision_threshold(
        args.min_precision_at_20 if args.min_precision_at_20 is not None else args.precision_threshold
    )
    evaluate_dataset(
        args.base_model,
        args.adapter_dir,
        args.data_dir,
        args.threshold,
        precision_top_k,
        precision_threshold,
        args.max_samples,
        max(64, args.max_seq_length or default_max_seq_length()),
        args.output_dir,
        args.min_positive_recall,
        args.down6_score_floor,
    )


if __name__ == "__main__":
    main()
