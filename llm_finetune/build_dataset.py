from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from a_share_crawler import DEFAULT_KTYPE, mysql_connect
from kline_statistics import SHORT_TERM_SURGE_TYPE, ensure_kline_statistics_table
from llm_finetune.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_WINDOW,
    collect_window_samples,
    parse_date_arg,
    to_messages_jsonl,
    write_jsonl,
)
from surge_pattern_miner import backfill_kline_stat_selection_dates

DEFAULT_VALID_RATIO = 0.2
DEFAULT_NEGATIVE_RATIO = 1.0
DEFAULT_BATCH_SIZE = 40
DEFAULT_SPLIT_SEED = 20260512


def stable_rank(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def split_rows(rows: list, valid_ratio: float, seed: int) -> tuple[list, list]:
    if not rows:
        return [], []
    ratio = min(max(valid_ratio, 0.01), 0.99)
    ordered = sorted(rows, key=lambda row: stable_rank(seed, row.scode, row.trade_date, row.label))
    valid_count = max(1, int(round(len(ordered) * ratio))) if len(ordered) > 1 else 0
    valid = ordered[:valid_count]
    train = ordered[valid_count:]
    return train, valid


def load_positive_events(conn, stat_type: str, start_date: date | None, end_date: date | None, limit: int | None) -> pd.DataFrame:
    sql = """
        SELECT SCode, COALESCE(SelectionDate, PrevTradeDate) AS TradeDate, GainRate
        FROM klinestatistics
        WHERE StatType = %s
          AND PrevTradeDate IS NOT NULL
    """
    params: list = [stat_type]
    if start_date is not None:
        sql += " AND COALESCE(SelectionDate, PrevTradeDate) >= %s"
        params.append(start_date)
    if end_date is not None:
        sql += " AND COALESCE(SelectionDate, PrevTradeDate) <= %s"
        params.append(end_date)
    sql += " ORDER BY SCode, TradeDate"
    if limit is not None and limit > 0:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["SCode", "TradeDate", "GainRate"])
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    df["GainRate"] = pd.to_numeric(df["GainRate"], errors="coerce")
    return df


def load_negative_events(
    conn,
    stat_type: str,
    start_date: date | None,
    end_date: date | None,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    date_filter = ""
    params: list = [stat_type, DEFAULT_KTYPE]
    if start_date is not None:
        date_filter += " AND dk.KTime >= %s"
        params.append(start_date)
    if end_date is not None:
        date_filter += " AND dk.KTime < %s"
        params.append(end_date + timedelta(days=1))
    scan_limit = max(1000, limit * 200)
    params.append(scan_limit)
    sql = f"""
        SELECT dk.SCode, DATE(dk.KTime) AS TradeDate, 0.0 AS GainRate
        FROM dkandles dk
        LEFT JOIN klinestatistics ks
          ON ks.SCode = dk.SCode
         AND ks.StatType = %s
         AND COALESCE(ks.SelectionDate, ks.PrevTradeDate) = DATE(dk.KTime)
        WHERE dk.KType = %s
          {date_filter}
          AND ks.Id IS NULL
        ORDER BY dk.SCode, dk.KTime
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["SCode", "TradeDate", "GainRate"])
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    df["_Rank"] = df.apply(lambda row: stable_rank(seed, row["SCode"], row["TradeDate"], "negative"), axis=1)
    return df.sort_values("_Rank").drop(columns=["_Rank"]).head(limit).reset_index(drop=True)


def build_dataset(
    output_dir: Path,
    stat_type: str,
    start_date: date | None,
    end_date: date | None,
    positive_limit: int | None,
    negative_ratio: float,
    valid_ratio: float,
    seed: int,
    daily_window: int,
    weekly_window: int,
    batch_size: int,
) -> tuple[int, int]:
    with mysql_connect() as conn:
        ensure_kline_statistics_table(conn)
        backfill_kline_stat_selection_dates(conn, stat_type)
        positive_events = load_positive_events(conn, stat_type, start_date, end_date, positive_limit)
        negative_limit = max(1, int(len(positive_events) * negative_ratio))
        negative_events = load_negative_events(conn, stat_type, start_date, end_date, negative_limit, seed)
        print(f"loaded events positives={len(positive_events)} negatives={len(negative_events)}", flush=True)
        positive_samples, positive_features = collect_window_samples(conn, positive_events, "positive", daily_window, weekly_window, batch_size)
        negative_samples, negative_features = collect_window_samples(conn, negative_events, "negative", daily_window, weekly_window, batch_size)

    allowed_features = positive_features | negative_features
    rows = [to_messages_jsonl(sample, allowed_features) for sample in positive_samples + negative_samples]
    train_rows, valid_rows = split_rows([sample for sample in positive_samples + negative_samples], valid_ratio, seed)
    train_json = [to_messages_jsonl(sample, allowed_features) for sample in train_rows]
    valid_json = [to_messages_jsonl(sample, allowed_features) for sample in valid_rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_count = write_jsonl(output_dir / "train.jsonl", train_json)
    valid_count = write_jsonl(output_dir / "valid.jsonl", valid_json)
    write_jsonl(output_dir / "all.jsonl", rows)
    (output_dir / "allowed_features.json").write_text(json.dumps({"features": sorted(allowed_features)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dataset written dir={output_dir} train={train_count} valid={valid_count} all={len(rows)} features={len(allowed_features)}", flush=True)
    return train_count, valid_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LoRA instruction-tuning data from dkandles/wkandles/klinestatistics")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stat-type", default=SHORT_TERM_SURGE_TYPE)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--positive-limit", type=int, help="Limit positives for a smoke dataset")
    parser.add_argument("--negative-ratio", type=float, default=DEFAULT_NEGATIVE_RATIO)
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--daily-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_dataset(
        output_dir=args.output_dir,
        stat_type=args.stat_type,
        start_date=parse_date_arg(args.start_date),
        end_date=parse_date_arg(args.end_date),
        positive_limit=args.positive_limit,
        negative_ratio=max(0.0, args.negative_ratio),
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        daily_window=max(2, args.daily_window),
        weekly_window=max(2, args.weekly_window),
        batch_size=max(1, args.batch_size),
    )


if __name__ == "__main__":
    main()
