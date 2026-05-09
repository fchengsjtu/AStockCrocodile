from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd

from a_share_crawler import DEFAULT_KTYPE, mysql_connect
from stock_selector import compute_strategy_frame, parse_date

DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_FORWARD_DAYS = 16


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str
    end_date: str
    min_turnover_amount: float
    limit_per_day: int | None
    ktype: str
    output: str | None


def load_backtest_daily(conn, start_date: date, end_date: date, ktype: str) -> pd.DataFrame:
    load_start = start_date - timedelta(days=DEFAULT_LOOKBACK_DAYS * 2)
    load_end = end_date + timedelta(days=DEFAULT_FORWARD_DAYS * 2)
    sql = """
        SELECT dk.SCode, si.SName, DATE(dk.KTime) AS TradeDate,
               dk.Open, dk.Close, dk.High, dk.Low, dk.Amount, dk.Volume,
               dk.MA5, dk.MA8, dk.MA13, dk.MA34, dk.MA55
        FROM dkandles dk
        LEFT JOIN stockinfo si ON si.SCode = dk.SCode
        WHERE dk.KType = %s AND dk.KTime >= %s AND dk.KTime < %s
        ORDER BY dk.SCode, dk.KTime
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ktype, load_start, load_end + timedelta(days=1)))
        rows = cur.fetchall()
    columns = ["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume", "MA5", "MA8", "MA13", "MA34", "MA55"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    for column in columns[3:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def weighted_average_price(window: pd.DataFrame) -> float | None:
    volume_sum = window["Volume"].sum(skipna=True)
    amount_sum = window["Amount"].sum(skipna=True)
    if pd.notna(volume_sum) and volume_sum > 0 and pd.notna(amount_sum) and amount_sum > 0:
        return float(amount_sum * 100 / volume_sum)
    typical = ((window["High"] + window["Low"] + window["Close"]) / 3).mean(skipna=True)
    if pd.isna(typical):
        return None
    return float(typical)


def build_trade_day_positions(daily_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for symbol, group in daily_df.groupby("SCode", sort=False):
        item = group.sort_values("TradeDate").reset_index(drop=True).copy()
        item["TradeIndex"] = range(len(item))
        result[symbol] = item
    return result


def evaluate_selection(row, trade_frames: dict[str, pd.DataFrame]) -> dict | None:
    symbol = row.SCode
    if symbol not in trade_frames:
        return None
    frame = trade_frames[symbol]
    matches = frame.index[frame["TradeDate"] == row.TradeDate].tolist()
    if not matches:
        return None
    pos = matches[0]
    window = frame.iloc[pos + 3 : pos + 9].copy()
    if len(window) < 6:
        return None

    close_price = float(row.Close)
    min_low = float(window["Low"].min(skipna=True))
    weighted_avg = weighted_average_price(window)
    success = min_low >= close_price * 1.02
    explosive = min_low >= close_price * 1.20
    failure = weighted_avg is not None and weighted_avg <= close_price * 0.99
    return {
        "TradeDate": row.TradeDate,
        "SCode": row.SCode,
        "SName": row.SName,
        "ClosePrice": close_price,
        "ForwardStart": window.iloc[0]["TradeDate"],
        "ForwardEnd": window.iloc[-1]["TradeDate"],
        "ForwardMinLow": min_low,
        "ForwardWeightedAvg": weighted_avg,
        "Success": success,
        "Failure": failure,
        "Explosive": explosive,
        "Score": row.Score,
        "Reason": row.Reason,
    }


def select_historical_signals(strategy_df: pd.DataFrame, start_date: date, end_date: date, limit_per_day: int | None) -> pd.DataFrame:
    selected = strategy_df[(strategy_df["TradeDate"] >= start_date) & (strategy_df["TradeDate"] <= end_date) & strategy_df["Selected"]].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(["TradeDate", "Score", "Amount"], ascending=[True, False, False])
    if limit_per_day is not None and limit_per_day > 0:
        selected = selected.groupby("TradeDate", group_keys=False).head(limit_per_day)
    return selected


def summarize_results(results: pd.DataFrame) -> dict:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "success_count": 0,
            "failure_count": 0,
            "explosive_count": 0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
            "explosive_rate": 0.0,
        }
    success_count = int(results["Success"].sum())
    failure_count = int(results["Failure"].sum())
    explosive_count = int(results["Explosive"].sum())
    return {
        "total": total,
        "success_count": success_count,
        "failure_count": failure_count,
        "explosive_count": explosive_count,
        "success_rate": success_count / total,
        "failure_rate": failure_count / total,
        "explosive_rate": explosive_count / total,
    }


def run_backtest(config: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    start_date = parse_date(config.start_date)
    end_date = parse_date(config.end_date)
    if start_date is None or end_date is None:
        raise ValueError("start-date and end-date are required")
    if start_date > end_date:
        raise ValueError("start-date must be <= end-date")

    with mysql_connect() as conn:
        daily_df = load_backtest_daily(conn, start_date, end_date, config.ktype)
    if daily_df.empty:
        results = pd.DataFrame()
        summary = summarize_results(results)
        return results, summary

    strategy_df = compute_strategy_frame(daily_df, min_turnover_amount=config.min_turnover_amount)
    selected = select_historical_signals(strategy_df, start_date, end_date, config.limit_per_day)
    trade_frames = build_trade_day_positions(daily_df)
    evaluated = []
    for row in selected.itertuples(index=False):
        item = evaluate_selection(row, trade_frames)
        if item is not None:
            evaluated.append(item)
    results = pd.DataFrame(evaluated)
    summary = summarize_results(results)
    if config.output:
        results.to_csv(config.output, index=False, encoding="utf-8-sig")
    return results, summary


def print_summary(summary: dict) -> None:
    print(f"total={summary['total']}")
    print(f"success={summary['success_count']} success_rate={summary['success_rate']:.2%}")
    print(f"failure={summary['failure_count']} failure_rate={summary['failure_rate']:.2%}")
    print(f"explosive={summary['explosive_count']} explosive_rate={summary['explosive_rate']:.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest A-share stock selection strategy")
    parser.add_argument("--start-date", required=True, help="Backtest start date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Backtest end date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--min-turnover-amount", type=float, default=0.0, help="Minimum 5-day average Amount for strategy")
    parser.add_argument("--limit-per-day", type=int, default=None, help="Maximum selected stocks per trade date")
    parser.add_argument("--ktype", default=DEFAULT_KTYPE)
    parser.add_argument("--output", help="Optional CSV path for detailed backtest rows")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, summary = run_backtest(
        BacktestConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            min_turnover_amount=args.min_turnover_amount,
            limit_per_day=args.limit_per_day,
            ktype=args.ktype,
            output=args.output,
        )
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
