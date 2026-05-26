from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_recall75.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_SAMPLE_MODE,
    DEFAULT_WINDOW,
    build_messages,
    DEFAULT_MAX_SEQ_LENGTH,
    default_max_seq_length,
    default_output_dir,
    iter_batches,
    load_kline_map,
    mysql_connect,
    parse_date,
    sample_mode_config,
    pick_weekly_window,
    pick_monthly_window,
    pick_window,
)
from blackbox_finetune_recall75.gpu import prepare_rtx3060
from blackbox_finetune_recall75.inference import load_model, score_prediction
from blackbox_finetune.prediction_store import save_top_predictions
from stock_selector import latest_trade_date
from surge_pattern_miner import load_symbols


def predict_day(
    base_model: str,
    adapter_dir: Path,
    trade_date,
    threshold: float,
    daily_window: int | None,
    weekly_window: int | None,
    monthly_window: int | None,
    sample_mode: str,
    batch_size: int,
    limit: int | None,
    output: Path | None,
    max_seq_length: int,
    save_db: bool,
    save_top_n: int,
) -> pd.DataFrame:
    anchor = parse_date(trade_date)
    model, tokenizer = load_model(base_model, adapter_dir)
    rows = []
    with mysql_connect() as conn:
        if trade_date is None:
            anchor = latest_trade_date(conn)
        symbols = load_symbols(conn, anchor, anchor)
        config = sample_mode_config(sample_mode)
        daily_count = daily_window or config["daily"]
        weekly_count = weekly_window or config["weekly"]
        monthly_count = config["monthly"] if monthly_window is None else monthly_window
        lookback_start = anchor - timedelta(days=max(750, monthly_count * 45, weekly_count * 14, daily_count * 5))
        batches = list(iter_batches(symbols, batch_size))
        print(f"blackbox recall75 predict date={anchor} symbols={len(symbols)} batches={len(batches)}", flush=True)
        for batch_index, batch in enumerate(batches, start=1):
            daily_map = load_kline_map(conn, "dkandles", "D", batch, lookback_start, anchor)
            weekly_map = load_kline_map(conn, "wkandles", "W", batch, lookback_start, anchor)
            monthly_map = load_kline_map(conn, "mkandles", "M", batch, lookback_start, anchor) if monthly_count > 0 else {}
            selected = 0
            for scode in batch:
                daily = pick_window(daily_map.get(scode, []), anchor, daily_count)
                weekly = pick_weekly_window(weekly_map.get(scode, []), daily_map.get(scode, []), anchor, weekly_count)
                monthly = pick_monthly_window(monthly_map.get(scode, []), anchor, monthly_count) if monthly_count > 0 else []
                if daily is None or weekly is None or monthly is None:
                    continue
                prompt = tokenizer.apply_chat_template(build_messages(scode, anchor, daily, weekly, monthly, sample_mode=sample_mode), tokenize=False, add_generation_prompt=True)
                pred = score_prediction(model, tokenizer, prompt, max_seq_length, threshold)
                if pred["label"] != "positive":
                    continue
                rows.append(
                    {
                        "TradeDate": anchor,
                        "SCode": scode,
                        "PositiveProbability": pred["positive_probability"],
                        "PositiveLoss": pred["positive_loss"],
                        "NegativeLoss": pred["negative_loss"],
                    }
                )
                selected += 1
            print(f"batch {batch_index}/{len(batches)} selected={selected}", flush=True)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("PositiveProbability", ascending=False)
        if limit and limit > 0:
            result = result.head(limit).copy()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, encoding="utf-8-sig")
    if save_db:
        with mysql_connect() as conn:
            saved = save_top_predictions(conn, result, "blackbox_finetune_recall75", threshold, max_seq_length, save_top_n)
        print(json.dumps({"date": str(anchor), "saved_predictions": saved, "strategy": "blackbox_finetune_recall75"}, ensure_ascii=False), flush=True)
    print(json.dumps({"date": str(anchor), "selected": len(result)}, ensure_ascii=False), flush=True)
    if not result.empty:
        print(result.to_string(index=False), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one trading day with recall75 black-box model")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=default_output_dir() / "adapter")
    parser.add_argument("--date", dest="trade_date", required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--sample-mode", choices=["short", "long", "xlong"], default=DEFAULT_SAMPLE_MODE)
    parser.add_argument("--daily-window", type=int, help="Override daily bars for the selected sample mode")
    parser.add_argument("--weekly-window", type=int, help="Override weekly bars for the selected sample mode")
    parser.add_argument("--monthly-window", type=int, help="Override monthly bars for the selected sample mode")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Override token length; default follows sample mode")
    parser.add_argument("--save-top-n", type=int, default=5, help="Save top N predictions to MySQL; default saves top 5.")
    parser.add_argument("--no-save-db", action="store_true", help="Do not save top predictions to MySQL.")
    parser.add_argument("--cuda-device", default="0", help="CUDA device id. Default binds the RTX3060 as cuda:0.")
    parser.add_argument("--allow-non-rtx3060", action="store_true", help="Allow CUDA devices whose name is not RTX 3060.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    predict_day(
        args.base_model,
        args.adapter_dir,
        args.trade_date,
        args.threshold,
        max(2, args.daily_window) if args.daily_window else None,
        max(2, args.weekly_window) if args.weekly_window else None,
        max(0, args.monthly_window) if args.monthly_window is not None else None,
        args.sample_mode,
        max(1, args.batch_size),
        args.limit,
        args.output,
        max(64, args.max_seq_length or default_max_seq_length(args.sample_mode)),
        not args.no_save_db,
        max(0, args.save_top_n),
    )


if __name__ == "__main__":
    main()
