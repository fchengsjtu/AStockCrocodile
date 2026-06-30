from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune.prediction_store import save_top_predictions
from blackbox_finetune_drop6 import common as drop_common
from blackbox_finetune_drop6.inference import load_model as load_drop_model
from blackbox_finetune_drop6.inference import score_prediction as score_drop_prediction
from blackbox_finetune_recall60 import common as up_common
from blackbox_finetune_recall60.gpu import prepare_rtx3060
from blackbox_finetune_recall60.inference import load_model as load_up_model
from blackbox_finetune_recall60.inference import score_prediction as score_up_prediction


def to_path(value: str) -> Path:
    text = str(value)
    if os.name != "nt" and len(text) >= 3 and text[1:3] in {":\\", ":/"}:
        text = f"/mnt/{text[0].lower()}/{text[3:].replace(chr(92), '/')}"
    return Path(text).expanduser()


def infer_base_model(adapter_dir: Path, base_model: str | None) -> str:
    if base_model:
        return base_model
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing adapter_config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved = config.get("base_model_name_or_path")
    if not resolved:
        raise ValueError(f"base model is missing in {config_path}; pass --base-model")
    return str(resolved)


def strategy_name_from_weight(drop_weight: float) -> str:
    label = ("%g" % drop_weight).replace(".", "_").replace("-", "m")
    return f"joint_up_drop_w{label}"


def load_symbols(conn, trade_date, ktype: str = "D") -> list[str]:
    sql = """
        SELECT DISTINCT SCode
        FROM dkandles
        WHERE KType = %s AND KTime >= %s AND KTime < %s
        ORDER BY SCode
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ktype, trade_date, trade_date + timedelta(days=1)))
        rows = cur.fetchall()
    return [str(row[0]) for row in rows]


def latest_trade_date(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(DATE(KTime)) FROM dkandles WHERE KType = 'D'")
        row = cur.fetchone()
    if not row or row[0] is None:
        raise RuntimeError("No daily K-line data found in dkandles.")
    return up_common.parse_date(row[0])


def build_prompt(row_messages: list[dict], tokenizer, system_prompt: str) -> str:
    prompt_messages = [dict(message) for message in row_messages]
    if prompt_messages:
        prompt_messages[0]["content"] = system_prompt
    return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)


def clear_cuda_cache() -> None:
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        gc.collect()


def score_up_all_symbols(
    *,
    conn,
    model,
    tokenizer,
    trade_date,
    symbols: list[str],
    sample_mode: str,
    daily_window: int | None,
    weekly_window: int | None,
    monthly_window: int | None,
    batch_size: int,
    max_seq_length: int,
) -> tuple[list[dict], int]:
    config = up_common.sample_mode_config(sample_mode)
    daily_count = daily_window or config["daily"]
    weekly_count = weekly_window or config["weekly"]
    monthly_count = config["monthly"] if monthly_window is None else monthly_window
    lookback_start = trade_date - timedelta(days=max(750, monthly_count * 45, weekly_count * 14, daily_count * 5))
    rows: list[dict] = []
    skipped_by_sample_rule = 0
    batches = list(up_common.iter_batches(symbols, batch_size))
    print(f"joint predict up scoring date={trade_date} symbols={len(symbols)} batches={len(batches)}", flush=True)
    for batch_index, batch in enumerate(batches, start=1):
        daily_map = up_common.load_kline_map(conn, "dkandles", "D", batch, lookback_start, trade_date)
        weekly_map = up_common.load_kline_map(conn, "wkandles", "W", batch, lookback_start, trade_date)
        monthly_map = up_common.load_kline_map(conn, "mkandles", "M", batch, lookback_start, trade_date) if monthly_count > 0 else {}
        scored = 0
        skipped = 0
        for scode in batch:
            daily = up_common.pick_window(daily_map.get(scode, []), trade_date, daily_count)
            weekly = up_common.pick_weekly_window(weekly_map.get(scode, []), daily_map.get(scode, []), trade_date, weekly_count)
            monthly = up_common.pick_monthly_window(monthly_map.get(scode, []), trade_date, monthly_count) if monthly_count > 0 else []
            if daily is None or weekly is None or monthly is None:
                continue
            if not up_common._sample_windows_are_valid(sample_mode, weekly, monthly, daily):
                skipped += 1
                skipped_by_sample_rule += 1
                continue
            messages = up_common.build_messages(scode, trade_date, daily, weekly, monthly, sample_mode=sample_mode)
            prompt = build_prompt(messages, tokenizer, up_common.SYSTEM_PROMPT)
            pred = score_up_prediction(model, tokenizer, prompt, max_seq_length, threshold=0.5)
            rows.append(
                {
                    "TradeDate": trade_date,
                    "SCode": scode,
                    "UpProbability": float(pred["positive_probability"]),
                    "UpPositiveLoss": float(pred["positive_loss"]),
                    "UpNegativeLoss": float(pred["negative_loss"]),
                    "messages": messages,
                }
            )
            scored += 1
        print(
            f"up batch {batch_index}/{len(batches)} scored={scored} skipped_by_sample_rule={skipped}",
            flush=True,
        )
    return rows, skipped_by_sample_rule


def score_drop_candidates(
    *,
    candidates: list[dict],
    model,
    tokenizer,
    max_seq_length: int,
) -> list[dict]:
    result: list[dict] = []
    for index, row in enumerate(candidates, start=1):
        prompt = build_prompt(row["messages"], tokenizer, drop_common.SYSTEM_PROMPT)
        pred = score_drop_prediction(model, tokenizer, prompt, max_seq_length, threshold=0.5)
        result.append(
            {
                **row,
                "DropProbability": float(pred["positive_probability"]),
                "DropPositiveLoss": float(pred["positive_loss"]),
                "DropNegativeLoss": float(pred["negative_loss"]),
            }
        )
        print(
            f"drop score {index}/{len(candidates)} scode={row['SCode']} "
            f"up={row['UpProbability']:.6f} drop={float(pred['positive_probability']):.6f}",
            flush=True,
        )
    return result


def predict_joint_day(
    *,
    base_model: str | None,
    up_adapter_dir: Path,
    drop_adapter_dir: Path,
    trade_date,
    sample_mode: str,
    batch_size: int,
    candidate_top_n: int,
    output_top_n: int,
    drop_weight: float,
    max_seq_length: int,
    output: Path | None,
    save_db: bool,
    strategy_name: str,
) -> pd.DataFrame:
    anchor = up_common.parse_date(trade_date) if trade_date is not None else None
    resolved_base_model = infer_base_model(up_adapter_dir, base_model)
    scanned_symbols = remaining_symbols = filtered_abnormal = skipped_by_sample_rule = 0

    with up_common.mysql_connect() as conn:
        if anchor is None:
            anchor = latest_trade_date(conn)
        symbols = load_symbols(conn, anchor)
        scanned_symbols = len(symbols)
        abnormal_symbols = up_common.load_abnormal_symbols(conn, symbols, anchor)
        filtered_abnormal = len(abnormal_symbols)
        if abnormal_symbols:
            symbols = [scode for scode in symbols if scode not in abnormal_symbols]
        remaining_symbols = len(symbols)
        print(
            f"joint predict date={anchor} scanned={scanned_symbols} "
            f"filtered_abnormal={filtered_abnormal} remaining={remaining_symbols}",
            flush=True,
        )
        up_model, up_tokenizer = load_up_model(resolved_base_model, up_adapter_dir)
        up_rows, skipped_by_sample_rule = score_up_all_symbols(
            conn=conn,
            model=up_model,
            tokenizer=up_tokenizer,
            trade_date=anchor,
            symbols=symbols,
            sample_mode=sample_mode,
            daily_window=None,
            weekly_window=None,
            monthly_window=None,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )
        del up_model
        del up_tokenizer
        clear_cuda_cache()

    up_top = sorted(up_rows, key=lambda row: row["UpProbability"], reverse=True)[:candidate_top_n]
    print(f"up scored={len(up_rows)} drop_candidates={len(up_top)}", flush=True)
    drop_model, drop_tokenizer = load_drop_model(resolved_base_model, drop_adapter_dir)
    joint_rows = score_drop_candidates(
        candidates=up_top,
        model=drop_model,
        tokenizer=drop_tokenizer,
        max_seq_length=max_seq_length,
    )
    del drop_model
    del drop_tokenizer
    clear_cuda_cache()

    for row in joint_rows:
        row["CombinedScore"] = float(row["UpProbability"]) - drop_weight * float(row["DropProbability"])
    joint_rows.sort(key=lambda row: row["CombinedScore"], reverse=True)
    result = pd.DataFrame(
        [
            {
                "TradeDate": row["TradeDate"],
                "SCode": row["SCode"],
                "CombinedScore": row["CombinedScore"],
                "UpProbability": row["UpProbability"],
                "DropProbability": row["DropProbability"],
                "UpPositiveLoss": row["UpPositiveLoss"],
                "UpNegativeLoss": row["UpNegativeLoss"],
                "DropPositiveLoss": row["DropPositiveLoss"],
                "DropNegativeLoss": row["DropNegativeLoss"],
            }
            for row in joint_rows[:output_top_n]
        ]
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, encoding="utf-8-sig")

    saved = 0
    if save_db:
        db_frame = pd.DataFrame(
            [
                {
                    "TradeDate": row["TradeDate"],
                    "SCode": row["SCode"],
                    "PositiveProbability": row["CombinedScore"],
                    "PositiveLoss": row["UpProbability"],
                    "NegativeLoss": row["DropProbability"],
                }
                for row in joint_rows
            ]
        )
        with up_common.mysql_connect() as conn:
            saved = save_top_predictions(
                conn,
                db_frame,
                strategy_name,
                threshold=drop_weight,
                max_seq_length=max_seq_length,
                top_n=output_top_n,
            )
    summary = {
        "date": str(anchor),
        "strategy": strategy_name,
        "up_scored": len(up_rows),
        "drop_candidates": len(up_top),
        "output_top_n": len(result),
        "saved_predictions": saved,
        "drop_weight": drop_weight,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not result.empty:
        print(result.to_string(index=False), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint up/drop binary prediction for one trading day.")
    parser.add_argument("--base-model", help="Base model override. Defaults to up adapter_config.json.")
    parser.add_argument("--up-adapter-dir", type=to_path, required=True)
    parser.add_argument("--drop-adapter-dir", type=to_path, required=True)
    parser.add_argument("--date", dest="trade_date", required=True)
    parser.add_argument("--sample-mode", choices=["short", "long", "xlong", "xxlong"], default="xlong")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--candidate-top-n", type=int, default=50)
    parser.add_argument("--output-top-n", type=int, default=20)
    parser.add_argument("--drop-weight", type=float, default=1.0)
    parser.add_argument("--strategy-name", help="Defaults to joint_up_drop_w<drop_weight>.")
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--output", type=to_path)
    parser.add_argument("--no-save-db", action="store_true")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--allow-non-rtx3060", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    predict_joint_day(
        base_model=args.base_model,
        up_adapter_dir=args.up_adapter_dir,
        drop_adapter_dir=args.drop_adapter_dir,
        trade_date=args.trade_date,
        sample_mode=args.sample_mode,
        batch_size=max(1, args.batch_size),
        candidate_top_n=max(1, args.candidate_top_n),
        output_top_n=max(1, args.output_top_n),
        drop_weight=args.drop_weight,
        max_seq_length=max(64, args.max_seq_length),
        output=args.output,
        save_db=not args.no_save_db,
        strategy_name=args.strategy_name or strategy_name_from_weight(args.drop_weight),
    )


if __name__ == "__main__":
    main()
