from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fingpt_forecaster_qlora.common import DEFAULT_BASE_MODEL, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def evaluate(base_model: str, adapter_dir: Path, data_dir: Path, threshold: float, max_samples: int | None) -> dict:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError("missing inference dependencies; install fingpt_forecaster_qlora/requirements.txt") from exc

    rows = []
    valid_path = data_dir / "valid.jsonl"
    with valid_path.open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, quantization_config=quantization_config)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    tp = fp = tn = fn = 0
    for row in rows:
        messages = row["messages"][:2]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        parsed = extract_json(text)
        predicted_positive = parsed.get("label") == "positive" and float(parsed.get("success_probability", 0.0) or 0.0) >= threshold
        actual_positive = int(row["metadata"]["label"]) == 1
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {"samples": len(rows), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "threshold": threshold}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a FinGPT-Forecaster QLoRA adapter")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "adapter")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--max-samples", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate(args.base_model, args.adapter_dir, args.data_dir, args.threshold, args.max_samples)


if __name__ == "__main__":
    main()
