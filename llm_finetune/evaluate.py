from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_finetune.common import BASE_MODEL, DEFAULT_DATA_DIR, DEFAULT_MIN_SUCCESS_RATE, DEFAULT_OUTPUT_DIR, read_jsonl


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def missing_adapter_error(adapter_dir: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"missing adapter_config.json under {adapter_dir}; train first:\n"
        "  python -m llm_finetune.train --base-model Qwen/Qwen2.5-0.5B-Instruct "
        "--data-dir llm_finetune/data --output-dir llm_finetune/runs/qwen2.5-0.5b-stock-lora"
    )


def candidate_answer(label: str, probability: float) -> str:
    reason = "Historical K-line setup matches trained positive surge samples." if label == "positive" else "Historical K-line setup does not match trained positive surge samples."
    return json.dumps({"label": label, "success_probability": probability, "reason": reason}, ensure_ascii=False, separators=(",", ":"))


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
    positive_loss = answer_loss(model, tokenizer, prompt, candidate_answer("positive", 0.8))
    negative_loss = answer_loss(model, tokenizer, prompt, candidate_answer("negative", 0.1))
    positive_weight = math.exp(-positive_loss)
    negative_weight = math.exp(-negative_loss)
    probability = positive_weight / (positive_weight + negative_weight) if positive_weight + negative_weight else 0.0
    return {
        "label": "positive" if probability >= 0.5 else "negative",
        "success_probability": probability,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
    }


def evaluate(base_model: str, adapter_dir: Path, data_dir: Path, threshold: float, min_success_rate: float, max_samples: int | None) -> dict:
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"missing test dataset: {test_path}")
    if not (adapter_dir / "adapter_config.json").exists():
        raise missing_adapter_error(adapter_dir)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("missing inference dependencies; run scripts/one_click_deploy first") from exc

    rows = read_jsonl(test_path, max_samples)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    tp = fp = tn = fn = predicted = 0
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        parsed = score_prediction(model, tokenizer, prompt)
        is_pred_positive = parsed.get("label") == "positive" and float(parsed.get("success_probability", 0) or 0) >= threshold
        is_actual_positive = int(row["metadata"]["label"]) == 1
        predicted += int(is_pred_positive)
        if is_pred_positive and is_actual_positive:
            tp += 1
        elif is_pred_positive and not is_actual_positive:
            fp += 1
        elif not is_pred_positive and is_actual_positive:
            fn += 1
        else:
            tn += 1
    success_rate = tp / predicted if predicted else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {
        "samples": len(rows),
        "predicted_positive": predicted,
        "success_rate": success_rate,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
        "min_success_rate": min_success_rate,
        "passed": success_rate >= min_success_rate if predicted else False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(f"evaluation failed: success_rate={success_rate:.4f} < {min_success_rate:.4f} or no positive predictions")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned Qwen stock pattern adapter")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "adapter")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--min-success-rate", type=float, default=DEFAULT_MIN_SUCCESS_RATE)
    parser.add_argument("--max-samples", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate(args.base_model, args.adapter_dir, args.data_dir, args.threshold, args.min_success_rate, args.max_samples)


if __name__ == "__main__":
    main()
