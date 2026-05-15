from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_finetune.common import (
    BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WINDOW,
    build_messages,
    load_kline_map,
    mysql_connect,
    parse_date,
    pick_window,
)
from llm_finetune.evaluate import missing_adapter_error, score_prediction


def predict(base_model: str, adapter_dir: Path, scode: str, anchor_date, daily_window: int, weekly_window: int) -> dict:
    anchor = parse_date(anchor_date)
    if not (adapter_dir / "adapter_config.json").exists():
        raise missing_adapter_error(adapter_dir)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("missing inference dependencies; run scripts/one_click_deploy first") from exc

    lookback_start = anchor - timedelta(days=max(500, weekly_window * 10, daily_window * 4))
    with mysql_connect() as conn:
        daily_map = load_kline_map(conn, "dkandles", "D", [scode], lookback_start, anchor)
        weekly_map = load_kline_map(conn, "wkandles", "W", [scode], lookback_start, anchor)
    daily = pick_window(daily_map.get(scode, []), anchor, daily_window)
    weekly = pick_window(weekly_map.get(scode, []), anchor, weekly_window)
    if daily is None or weekly is None:
        raise RuntimeError(f"not enough K-line history for {scode} at {anchor}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    prompt = tokenizer.apply_chat_template(build_messages(scode, anchor, daily, weekly), tokenize=False, add_generation_prompt=True)
    result = score_prediction(model, tokenizer, prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one stock/date with the fine-tuned Qwen adapter")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "adapter")
    parser.add_argument("--scode", required=True)
    parser.add_argument("--date", required=True, help="Anchor date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--daily-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WINDOW)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    predict(args.base_model, args.adapter_dir, args.scode, args.date, args.daily_window, args.weekly_window)


if __name__ == "__main__":
    main()
