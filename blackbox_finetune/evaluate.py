from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune.common import DEFAULT_BASE_MODEL, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, label_answer, read_jsonl
from llm_finetune.evaluate import missing_adapter_error


def answer_loss(model, tokenizer, prompt: str, answer: str) -> float:
    import torch

    answer = answer + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_ids.shape[1] :] = answer_ids
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels)
    return float(output.loss.detach().cpu())


def score_prediction(model, tokenizer, prompt: str) -> dict:
    positive_loss = answer_loss(model, tokenizer, prompt, label_answer(1))
    negative_loss = answer_loss(model, tokenizer, prompt, label_answer(0))
    positive_weight = math.exp(-positive_loss)
    negative_weight = math.exp(-negative_loss)
    probability = positive_weight / (positive_weight + negative_weight) if positive_weight + negative_weight else 0.0
    return {
        "label": "positive" if probability >= 0.5 else "negative",
        "positive_probability": probability,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
    }


def load_model(base_model: str, adapter_dir: Path):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("missing inference dependencies; run one-click deploy first") from exc
    if not (adapter_dir / "adapter_config.json").exists():
        raise missing_adapter_error(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def evaluate_rows(base_model: str, adapter_dir: Path, rows: list[dict], threshold: float) -> dict:
    model, tokenizer = load_model(base_model, adapter_dir)
    tp = fp = tn = fn = positives = positive_correct = 0
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        pred = score_prediction(model, tokenizer, prompt)
        predicted_positive = pred["label"] == "positive" and pred["positive_probability"] >= threshold
        actual_positive = int(row["metadata"]["label"]) == 1
        if actual_positive:
            positives += 1
        if predicted_positive and actual_positive:
            tp += 1
            positive_correct += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1
    positive_recall = positive_correct / positives if positives else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "samples": len(rows),
        "positive_samples": positives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_recall": positive_recall,
        "precision": precision,
        "threshold": threshold,
    }


def evaluate_dataset(
    base_model: str,
    adapter_dir: Path,
    data_dir: Path,
    threshold: float,
    min_positive_recall: float,
    max_samples: int | None,
) -> dict:
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"missing test dataset: {test_path}")
    rows = read_jsonl(test_path, max_samples)
    result = evaluate_rows(base_model, adapter_dir, rows, threshold)
    result["min_positive_recall"] = min_positive_recall
    result["passed"] = result["positive_recall"] >= min_positive_recall
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(f"positive_recall={result['positive_recall']:.4f} < {min_positive_recall:.4f}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate black-box fine-tuned classifier")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "adapter")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-positive-recall", type=float, default=0.60)
    parser.add_argument("--max-samples", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate_dataset(args.base_model, args.adapter_dir, args.data_dir, args.threshold, args.min_positive_recall, args.max_samples)


if __name__ == "__main__":
    main()

