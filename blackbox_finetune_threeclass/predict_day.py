from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from blackbox_finetune_recall60.common import _sample_windows_are_valid
from blackbox_finetune_threeclass.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLE_MODE,
    build_messages,
    iter_batches,
    load_abnormal_symbols,
    load_kline_map,
    mysql_connect,
    parse_date,
    pick_monthly_window,
    pick_weekly_window,
    pick_window,
    sample_mode_config,
)
from blackbox_finetune_threeclass.gpu import prepare_rtx3060
from blackbox_finetune_threeclass.inference import load_model, score_prediction


def _load_symbols(conn, anchor) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT SCode
            FROM dkandles
            WHERE KType = 'D' AND KTime >= %s AND KTime < %s
            ORDER BY SCode
            """,
            (anchor, anchor + timedelta(days=1)),
        )
        return [str(row[0]) for row in cur.fetchall()]


def predict_day(
    base_model: str,
    adapter_dir: Path,
    trade_date,
    sample_mode: str,
    batch_size: int,
    limit: int,
    positive_threshold: float,
    max_seq_length: int,
    output: Path | None,
) -> pd.DataFrame:
    anchor = parse_date(trade_date)
    model, tokenizer = load_model(base_model, adapter_dir)
    config = sample_mode_config(sample_mode)
    daily_count = config["daily"]
    weekly_count = config["weekly"]
    monthly_count = config["monthly"]
    lookback_start = anchor - timedelta(days=max(750, monthly_count * 45, weekly_count * 14, daily_count * 5))
    predictions = []
    with mysql_connect() as conn:
        symbols = _load_symbols(conn, anchor)
        abnormal = load_abnormal_symbols(conn, symbols, anchor)
        symbols = [scode for scode in symbols if scode not in abnormal]
        batches = list(iter_batches(symbols, batch_size))
        print(f"threeclass predict date={anchor} symbols={len(symbols)} batches={len(batches)}", flush=True)
        for batch_index, batch in enumerate(batches, start=1):
            daily_map = load_kline_map(conn, "dkandles", "D", batch, lookback_start, anchor)
            weekly_map = load_kline_map(conn, "wkandles", "W", batch, lookback_start, anchor)
            monthly_map = load_kline_map(conn, "mkandles", "M", batch, lookback_start, anchor) if monthly_count else {}
            scored = 0
            for scode in batch:
                daily = pick_window(daily_map.get(scode, []), anchor, daily_count)
                weekly = pick_weekly_window(weekly_map.get(scode, []), daily_map.get(scode, []), anchor, weekly_count)
                monthly = pick_monthly_window(monthly_map.get(scode, []), anchor, monthly_count) if monthly_count else []
                if daily is None or weekly is None or monthly is None:
                    continue
                if not _sample_windows_are_valid(sample_mode, weekly, monthly, daily):
                    continue
                prompt = tokenizer.apply_chat_template(
                    build_messages(scode, anchor, daily, weekly, monthly, sample_mode=sample_mode),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prediction = score_prediction(model, tokenizer, prompt, max_seq_length)
                scored += 1
                if prediction["positive_probability"] < positive_threshold:
                    continue
                predictions.append(
                    {
                        "TradeDate": anchor,
                        "SCode": scode,
                        "PredictedClass": prediction["label"],
                        "PositiveProbability": prediction["positive_probability"],
                        "NegativeProbability": prediction["negative_probability"],
                        "NeutralProbability": prediction["neutral_probability"],
                    }
                )
            print(f"batch {batch_index}/{len(batches)} scored={scored}", flush=True)
    result = pd.DataFrame(predictions)
    if not result.empty:
        result = result.sort_values("PositiveProbability", ascending=False).head(max(1, limit)).copy()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, encoding="utf-8-sig")
    if not result.empty:
        print(result.to_string(index=False), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one day with the three-class black-box model")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "adapter")
    parser.add_argument("--date", required=True)
    parser.add_argument("--sample-mode", choices=["short", "long", "xlong", "xxlong"], default=DEFAULT_SAMPLE_MODE)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--positive-threshold", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--allow-non-rtx3060", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    predict_day(
        args.base_model,
        args.adapter_dir,
        args.date,
        args.sample_mode,
        max(1, args.batch_size),
        max(1, args.limit),
        min(max(args.positive_threshold, 0.0), 1.0),
        max(64, args.max_seq_length),
        args.output,
    )


if __name__ == "__main__":
    main()

