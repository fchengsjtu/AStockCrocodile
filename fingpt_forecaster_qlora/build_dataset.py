from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fingpt_forecaster_qlora.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_STAT_TYPE,
    ForecastSample,
    compact_kline_rows,
    load_kline_window,
    load_nearby_news,
    mysql_connect,
    parse_date,
    sample_to_messages,
    write_jsonl,
)


def stable_rank(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def load_positive_events(conn, stat_type: str, start_date, end_date, limit: int | None) -> pd.DataFrame:
    sql = """
        SELECT SCode, COALESCE(SelectionDate, PrevTradeDate) AS TradeDate, GainRate
        FROM klinestatistics
        WHERE StatType = %s
          AND COALESCE(SelectionDate, PrevTradeDate) IS NOT NULL
    """
    params: list = [stat_type]
    if start_date is not None:
        sql += " AND COALESCE(SelectionDate, PrevTradeDate) >= %s"
        params.append(start_date)
    if end_date is not None:
        sql += " AND COALESCE(SelectionDate, PrevTradeDate) <= %s"
        params.append(end_date)
    sql += " ORDER BY SCode, TradeDate"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["SCode", "TradeDate", "GainRate"])
    if not df.empty:
        df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
        df["GainRate"] = pd.to_numeric(df["GainRate"], errors="coerce").fillna(0.0)
    return df


def load_negative_events(conn, stat_type: str, start_date, end_date, limit: int, seed: int) -> pd.DataFrame:
    date_filter = ""
    params: list = [stat_type, "D"]
    if start_date is not None:
        date_filter += " AND dk.KTime >= %s"
        params.append(start_date)
    if end_date is not None:
        date_filter += " AND dk.KTime < %s"
        params.append(pd.Timestamp(end_date) + pd.Timedelta(days=1))
    params.append(max(1000, limit * 200))
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


def split_rows(rows: list[dict], valid_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []
    ratio = min(max(valid_ratio, 0.01), 0.99)
    ordered = sorted(rows, key=lambda row: stable_rank(seed, row["metadata"]["scode"], row["metadata"]["trade_date"], row["metadata"]["label"]))
    valid_count = max(1, int(round(len(ordered) * ratio))) if len(ordered) > 1 else 0
    return ordered[valid_count:], ordered[:valid_count]


def collect_samples(conn, events: pd.DataFrame, label: int, daily_window: int, weekly_window: int, news_days: int, min_success_rate: float) -> list[dict]:
    rows: list[dict] = []
    for index, event in enumerate(events.itertuples(index=False), start=1):
        daily = load_kline_window(conn, "dkandles", "D", event.SCode, event.TradeDate, daily_window)
        weekly = load_kline_window(conn, "wkandles", "W", event.SCode, event.TradeDate, weekly_window)
        if len(daily) < daily_window or len(weekly) < weekly_window:
            continue
        sample = ForecastSample(
            scode=event.SCode,
            trade_date=event.TradeDate,
            label=label,
            gain_rate=float(event.GainRate or 0.0),
            daily_rows=compact_kline_rows(daily),
            weekly_rows=compact_kline_rows(weekly),
            news_rows=load_nearby_news(conn, event.TradeDate, days=news_days),
        )
        rows.append(sample_to_messages(sample, min_success_rate))
        if index % 200 == 0:
            print(f"collected label={label} scanned={index} usable={len(rows)}", flush=True)
    return rows


def build_dataset(
    output_dir: Path,
    stat_type: str,
    start_date,
    end_date,
    positive_limit: int | None,
    negative_ratio: float,
    valid_ratio: float,
    seed: int,
    daily_window: int,
    weekly_window: int,
    news_days: int,
    min_success_rate: float,
) -> tuple[int, int]:
    with mysql_connect() as conn:
        positives = load_positive_events(conn, stat_type, start_date, end_date, positive_limit)
        negative_limit = max(1, int(len(positives) * max(0.0, negative_ratio)))
        negatives = load_negative_events(conn, stat_type, start_date, end_date, negative_limit, seed)
        print(f"events loaded positives={len(positives)} negatives={len(negatives)}", flush=True)
        positive_rows = collect_samples(conn, positives, 1, daily_window, weekly_window, news_days, min_success_rate)
        negative_rows = collect_samples(conn, negatives, 0, daily_window, weekly_window, news_days, min_success_rate)
    train_rows, valid_rows = split_rows(positive_rows + negative_rows, valid_ratio, seed)
    train_count = write_jsonl(output_dir / "train.jsonl", train_rows)
    valid_count = write_jsonl(output_dir / "valid.jsonl", valid_rows)
    all_count = write_jsonl(output_dir / "all.jsonl", train_rows + valid_rows)
    print(f"dataset written dir={output_dir} train={train_count} valid={valid_count} all={all_count}", flush=True)
    return train_count, valid_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FinGPT-Forecaster QLoRA JSONL dataset from MySQL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stat-type", default=DEFAULT_STAT_TYPE)
    parser.add_argument("--start-date", default="20100101")
    parser.add_argument("--end-date", default="20251231")
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--daily-window", type=int, default=55)
    parser.add_argument("--weekly-window", type=int, default=55)
    parser.add_argument("--news-days", type=int, default=3)
    parser.add_argument("--min-success-rate", type=float, default=0.40)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_dataset(
        output_dir=args.output_dir,
        stat_type=args.stat_type,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        positive_limit=args.positive_limit,
        negative_ratio=args.negative_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        daily_window=max(2, args.daily_window),
        weekly_window=max(2, args.weekly_window),
        news_days=max(0, args.news_days),
        min_success_rate=args.min_success_rate,
    )


if __name__ == "__main__":
    main()
