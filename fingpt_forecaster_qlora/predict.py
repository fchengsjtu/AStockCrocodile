from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fingpt_forecaster_qlora.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    ForecastSample,
    build_prompt,
    compact_kline_rows,
    load_kline_window,
    load_nearby_news,
    mysql_connect,
    parse_date,
    SYSTEM_PROMPT,
)


def predict(base_model: str, adapter_dir: Path, scode: str, trade_date, daily_window: int, weekly_window: int) -> str:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError("missing inference dependencies; install fingpt_forecaster_qlora/requirements.txt") from exc

    with mysql_connect() as conn:
        daily = load_kline_window(conn, "dkandles", "D", scode, trade_date, daily_window)
        weekly = load_kline_window(conn, "wkandles", "W", scode, trade_date, weekly_window)
        if len(daily) < daily_window or len(weekly) < weekly_window:
            raise RuntimeError(f"not enough kline rows for {scode} {trade_date}: daily={len(daily)} weekly={len(weekly)}")
        sample = ForecastSample(
            scode=scode,
            trade_date=trade_date,
            label=0,
            gain_rate=0.0,
            daily_rows=compact_kline_rows(daily),
            weekly_rows=compact_kline_rows(weekly),
            news_rows=load_nearby_news(conn, trade_date),
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, quantization_config=quantization_config)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(sample)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(text, flush=True)
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one stock/date with the trained FinGPT-Forecaster QLoRA adapter")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "adapter")
    parser.add_argument("--scode", required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--daily-window", type=int, default=55)
    parser.add_argument("--weekly-window", type=int, default=55)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    predict(args.base_model, args.adapter_dir, args.scode, parse_date(args.trade_date), args.daily_window, args.weekly_window)


if __name__ == "__main__":
    main()
