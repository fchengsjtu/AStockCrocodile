from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from a_share_crawler import mysql_connect
from kline_statistics import SHORT_TERM_SURGE_TYPE
from llm_finetune.common import DEFAULT_BASE_MODEL, DEFAULT_DATA_DIR, DEFAULT_FINETUNE_DIR
from llm_surge_pattern_miner import parse_llm_patterns
from surge_pattern_miner import (
    SurgePatternConfig,
    ensure_surge_pattern_table,
    evaluate_patterns,
    load_positive_events,
    save_patterns,
)

DEFAULT_MIN_SUCCESS_RATE = 0.40
DEFAULT_DAILY_WINDOW = 55
DEFAULT_WEEKLY_WINDOW = 55
DEFAULT_BATCH_SIZE = 40


def load_validation_prompts(data_dir: Path, limit: int | None) -> list[dict]:
    path = data_dir / "valid.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing validation dataset: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_allowed_features(data_dir: Path) -> set[str]:
    path = data_dir / "allowed_features.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item) for item in payload.get("features", [])}


def load_model_generator(base_model: str, adapter_dir: Path, max_new_tokens: int):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Missing inference dependencies. Install them in your GPU Python environment:\n"
            "pip install -r requirements-finetune.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir if adapter_dir.exists() else base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    def generate(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return generate


def collect_model_patterns(
    base_model: str,
    adapter_dir: Path,
    data_dir: Path,
    valid_limit: int | None,
    min_pattern_size: int,
    max_pattern_size: int,
    max_new_tokens: int,
) -> tuple[set[tuple[str, ...]], Counter, date | None, date | None]:
    allowed_features = load_allowed_features(data_dir)
    rows = load_validation_prompts(data_dir, valid_limit)
    generate = load_model_generator(base_model, adapter_dir, max_new_tokens)
    patterns: set[tuple[str, ...]] = set()
    support: Counter[tuple[str, ...]] = Counter()
    dates: list[date] = []
    for index, row in enumerate(rows, start=1):
        metadata = row.get("metadata", {})
        if metadata.get("trade_date"):
            dates.append(pd.to_datetime(metadata["trade_date"]).date())
        text = generate(row["messages"])
        parsed = parse_llm_patterns(text, allowed_features, min_pattern_size, max_pattern_size)
        for pattern in parsed:
            patterns.add(pattern)
            support[pattern] += 1
        print(f"evaluate adapter prompt {index}/{len(rows)} patterns_total={len(patterns)}", flush=True)
    return patterns, support, (min(dates) if dates else None), (max(dates) if dates else None)


def run_evaluation(
    base_model: str,
    adapter_dir: Path,
    data_dir: Path,
    stat_type: str,
    min_success_rate: float,
    min_sample_count: int,
    min_positive_support: int,
    min_pattern_size: int,
    max_pattern_size: int,
    valid_limit: int | None,
    daily_window: int,
    weekly_window: int,
    batch_size: int,
    max_new_tokens: int,
    output: Path | None,
    save_db: bool,
) -> pd.DataFrame:
    patterns, support, start_date, end_date = collect_model_patterns(
        base_model=base_model,
        adapter_dir=adapter_dir,
        data_dir=data_dir,
        valid_limit=valid_limit,
        min_pattern_size=min_pattern_size,
        max_pattern_size=max_pattern_size,
        max_new_tokens=max_new_tokens,
    )
    if start_date is None or end_date is None:
        raise RuntimeError("Validation data did not contain trade_date metadata")
    if not patterns:
        raise RuntimeError("Fine-tuned model did not generate any valid feature-token patterns")

    config = SurgePatternConfig(
        test_start_date=str(start_date),
        test_end_date=str(end_date),
        train_start_date=str(start_date),
        train_end_date=str(end_date),
        stat_type=stat_type,
        min_success_rates=(min_success_rate,),
        min_sample_count=min_sample_count,
        min_positive_support=min_positive_support,
        max_pattern_size=max_pattern_size,
        daily_window=daily_window,
        weekly_window=weekly_window,
        batch_size=batch_size,
        output=str(output) if output else None,
        save_db=save_db,
    )
    with mysql_connect() as conn:
        positives = load_positive_events(conn, stat_type, start_date, end_date)
        if not positives.empty:
            positives = positives.copy()
            positives["SelectionDate"] = positives["PrevTradeDate"]
        results = evaluate_patterns(conn, positives, patterns, support, config, start_date, end_date)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output, index=False, encoding="utf-8-sig")
        saved = 0
        if save_db:
            ensure_surge_pattern_table(conn)
            saved = save_patterns(conn, results, config, start_date, end_date, start_date, end_date)
    print(f"fine-tuned model patterns={len(patterns)} kept={len(results)} saved={saved}", flush=True)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA fine-tuned stock pattern model and save validated rules to surgepatterns")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_FINETUNE_DIR / "adapter")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stat-type", default=SHORT_TERM_SURGE_TYPE)
    parser.add_argument("--min-success-rate", type=float, default=DEFAULT_MIN_SUCCESS_RATE)
    parser.add_argument("--min-sample-count", type=int, default=20)
    parser.add_argument("--min-positive-support", type=int, default=1)
    parser.add_argument("--min-pattern-size", type=int, default=3)
    parser.add_argument("--max-pattern-size", type=int, default=8)
    parser.add_argument("--valid-limit", type=int, default=200)
    parser.add_argument("--daily-window", type=int, default=DEFAULT_DAILY_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WEEKLY_WINDOW)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("data") / "finetuned_surge_patterns.csv")
    parser.add_argument("--no-save-db", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    min_success_rate = args.min_success_rate / 100 if args.min_success_rate > 1 else args.min_success_rate
    run_evaluation(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        data_dir=args.data_dir,
        stat_type=args.stat_type,
        min_success_rate=min_success_rate,
        min_sample_count=max(1, args.min_sample_count),
        min_positive_support=max(1, args.min_positive_support),
        min_pattern_size=max(1, args.min_pattern_size),
        max_pattern_size=max(max(1, args.min_pattern_size), args.max_pattern_size),
        valid_limit=args.valid_limit,
        daily_window=max(2, args.daily_window),
        weekly_window=max(2, args.weekly_window),
        batch_size=max(1, args.batch_size),
        max_new_tokens=max(32, args.max_new_tokens),
        output=args.output,
        save_db=not args.no_save_db,
    )


if __name__ == "__main__":
    main()
