from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import pymysql

from a_share_crawler import DEFAULT_KTYPE, mysql_connect, none_if_nan
from stock_selector import (
    STRATEGIES,
    STRATEGY_MA_BULLISH,
    STRATEGY_NEWS_HOT,
    STRATEGY_WEEKLY_VOLUME_DROP,
    compute_news_hot_selection,
    compute_strategy_frame,
    compute_weekly_volume_drop_selection,
    load_news_for_date,
    parse_date,
)

DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_FORWARD_DAYS = 16
DEFAULT_STRATEGY_NAME = STRATEGY_MA_BULLISH
BACKTEST_RESULT_TABLE = "strategybacktestresults"
SYMBOL_BATCH_SIZE = 300


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str
    end_date: str
    min_turnover_amount: float
    limit_per_day: int | None
    ktype: str
    output: str | None
    strategy_name: str
    save_db: bool
    min_recommendations: int
    max_recommendations: int


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


def normalize_daily_frame(rows) -> pd.DataFrame:
    columns = ["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume", "MA5", "MA8", "MA13", "MA34", "MA55"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    for column in columns[3:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def iter_batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def load_symbols(conn, table_name: str, ktype: str, start_date: date, end_date: date) -> list[str]:
    sql = f"""
        SELECT DISTINCT SCode
        FROM {table_name}
        WHERE KType = %s AND KTime >= %s AND KTime < %s
        ORDER BY SCode
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ktype, start_date, end_date + timedelta(days=1)))
        rows = cur.fetchall()
    return [row[0] for row in rows]


def load_backtest_daily_for_symbols(conn, symbols: list[str], start_date: date, end_date: date, ktype: str) -> pd.DataFrame:
    if not symbols:
        return normalize_daily_frame([])
    frames = []
    load_start = start_date
    load_end = end_date + timedelta(days=DEFAULT_FORWARD_DAYS * 2)
    for batch in iter_batches(symbols, SYMBOL_BATCH_SIZE):
        placeholders = ",".join(["%s"] * len(batch))
        sql = f"""
            SELECT dk.SCode, si.SName, DATE(dk.KTime) AS TradeDate,
                   dk.Open, dk.Close, dk.High, dk.Low, dk.Amount, dk.Volume,
                   dk.MA5, dk.MA8, dk.MA13, dk.MA34, dk.MA55
            FROM dkandles dk
            LEFT JOIN stockinfo si ON si.SCode = dk.SCode
            WHERE dk.KType = %s
              AND dk.SCode IN ({placeholders})
              AND dk.KTime >= %s AND dk.KTime < %s
            ORDER BY dk.SCode, dk.KTime
        """
        params = [ktype, *batch, load_start, load_end + timedelta(days=1)]
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        frame = normalize_daily_frame(rows)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return normalize_daily_frame([])
    return pd.concat(frames, ignore_index=True)


def load_backtest_weekly(conn, start_date: date, end_date: date) -> pd.DataFrame:
    load_start = start_date - timedelta(days=70)
    load_end = end_date + timedelta(days=7)
    sql = """
        SELECT wk.SCode, si.SName, DATE(wk.KTime) AS TradeDate,
               wk.Open, wk.Close, wk.High, wk.Low, wk.Amount, wk.Volume
        FROM wkandles wk
        LEFT JOIN stockinfo si ON si.SCode = wk.SCode
        WHERE wk.KType = 'W' AND wk.KTime >= %s AND wk.KTime < %s
        ORDER BY wk.SCode, wk.KTime
    """
    with conn.cursor() as cur:
        cur.execute(sql, (load_start, load_end + timedelta(days=1)))
        rows = cur.fetchall()
    columns = ["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    for column in columns[3:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_backtest_weekly_for_symbols(conn, symbols: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume"])
    load_start = start_date - timedelta(days=70)
    load_end = end_date + timedelta(days=7)
    frames = []
    for batch in iter_batches(symbols, SYMBOL_BATCH_SIZE):
        placeholders = ",".join(["%s"] * len(batch))
        sql = f"""
            SELECT wk.SCode, si.SName, DATE(wk.KTime) AS TradeDate,
                   wk.Open, wk.Close, wk.High, wk.Low, wk.Amount, wk.Volume
            FROM wkandles wk
            LEFT JOIN stockinfo si ON si.SCode = wk.SCode
            WHERE wk.KType = 'W'
              AND wk.SCode IN ({placeholders})
              AND wk.KTime >= %s AND wk.KTime < %s
            ORDER BY wk.SCode, wk.KTime
        """
        params = [*batch, load_start, load_end + timedelta(days=1)]
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        columns = ["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume"]
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            continue
        frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.date
        for column in columns[3:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume"])
    return pd.concat(frames, ignore_index=True)


def weighted_average_price(window: pd.DataFrame) -> float | None:
    volume_sum = window["Volume"].sum(skipna=True)
    amount_sum = window["Amount"].sum(skipna=True)
    if pd.notna(volume_sum) and volume_sum > 0 and pd.notna(amount_sum) and amount_sum > 0:
        return float(amount_sum * 100 / volume_sum)
    typical = ((window["High"] + window["Low"] + window["Close"]) / 3).mean(skipna=True)
    if pd.isna(typical):
        return None
    return float(typical)


def ensure_backtest_result_table(conn: pymysql.connections.Connection) -> None:
    sql = f"""
        CREATE TABLE IF NOT EXISTS {BACKTEST_RESULT_TABLE} (
            Id BIGINT NOT NULL AUTO_INCREMENT,
            SCode VARCHAR(10) NOT NULL,
            StrategyName VARCHAR(64) NOT NULL,
            StartDate DATE NOT NULL,
            EndDate DATE NOT NULL,
            SelectionDate DATE NULL,
            SampleCount INT NOT NULL,
            SuccessRate DECIMAL(18,6) NOT NULL,
            AvgRiseRate DECIMAL(18,6) NULL,
            FailureRate DECIMAL(18,6) NOT NULL,
            AvgDropRate DECIMAL(18,6) NULL,
            ExplosiveRate DECIMAL(18,6) NOT NULL,
            CreatedOn DATETIME NOT NULL,
            PRIMARY KEY (Id),
            UNIQUE KEY ux_strategybacktest_code_strategy_selection (SCode, StrategyName, StartDate, EndDate, SelectionDate),
            KEY idx_strategybacktest_strategy (StrategyName)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute(f"SHOW COLUMNS FROM {BACKTEST_RESULT_TABLE} LIKE 'SelectionDate'")
        if cur.fetchone() is None:
            cur.execute(f"ALTER TABLE {BACKTEST_RESULT_TABLE} ADD COLUMN SelectionDate DATE NULL AFTER EndDate")
        cur.execute(f"SHOW INDEX FROM {BACKTEST_RESULT_TABLE} WHERE Key_name = 'ux_strategybacktest_code_strategy_range'")
        if cur.fetchone() is not None:
            cur.execute(f"ALTER TABLE {BACKTEST_RESULT_TABLE} DROP INDEX ux_strategybacktest_code_strategy_range")
        cur.execute(f"SHOW INDEX FROM {BACKTEST_RESULT_TABLE} WHERE Key_name = 'ux_strategybacktest_code_strategy_selection'")
        if cur.fetchone() is None:
            cur.execute(
                f"ALTER TABLE {BACKTEST_RESULT_TABLE} "
                "ADD UNIQUE KEY ux_strategybacktest_code_strategy_selection (SCode, StrategyName, StartDate, EndDate, SelectionDate)"
            )
    conn.commit()


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
        matches = frame.index[frame["TradeDate"] >= row.TradeDate].tolist()
    if not matches:
        return None
    pos = matches[0]
    window = frame.iloc[pos + 3 : pos + 9].copy()
    if len(window) < 6:
        return None

    close_price = float(row.Close)
    min_low = float(window["Low"].min(skipna=True))
    weighted_avg = weighted_average_price(window)
    rise_rate = min_low / close_price - 1 if close_price > 0 else None
    weighted_drop_rate = weighted_avg / close_price - 1 if weighted_avg is not None and close_price > 0 else None
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
        "RiseRate": rise_rate,
        "WeightedDropRate": weighted_drop_rate,
        "Success": success,
        "Failure": failure,
        "Explosive": explosive,
        "Score": row.Score,
        "Reason": row.Reason,
        "StrategyName": getattr(row, "StrategyName", DEFAULT_STRATEGY_NAME),
    }


def select_historical_signals(strategy_df: pd.DataFrame, start_date: date, end_date: date, limit_per_day: int | None) -> pd.DataFrame:
    selected = strategy_df[(strategy_df["TradeDate"] >= start_date) & (strategy_df["TradeDate"] <= end_date) & strategy_df["Selected"]].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(["TradeDate", "Score", "Amount"], ascending=[True, False, False])
    if limit_per_day is not None and limit_per_day > 0:
        selected = selected.groupby("TradeDate", group_keys=False).head(limit_per_day)
    selected["StrategyName"] = STRATEGY_MA_BULLISH
    return selected


def build_ma_bullish_signals(daily_df: pd.DataFrame, start_date: date, end_date: date, config: BacktestConfig) -> pd.DataFrame:
    strategy_df = compute_strategy_frame(daily_df, min_turnover_amount=config.min_turnover_amount)
    selected = select_historical_signals(strategy_df, start_date, end_date, config.limit_per_day)
    if selected.empty:
        return selected
    return selected[["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"]]


def build_news_hot_signals(conn, daily_df: pd.DataFrame, start_date: date, end_date: date, config: BacktestConfig) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(columns=["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"])
    frames = []
    trade_dates = sorted(day for day in daily_df["TradeDate"].dropna().unique() if start_date <= day <= end_date)
    for trade_date in trade_dates:
        news_df = load_news_for_date(conn, trade_date)
        selected = compute_news_hot_selection(
            daily_df=daily_df,
            news_df=news_df,
            trade_date=trade_date,
            min_recommendations=config.min_recommendations,
            max_recommendations=config.max_recommendations,
        )
        if config.limit_per_day is not None and config.limit_per_day > 0:
            selected = selected.head(config.limit_per_day)
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"])
    return pd.concat(frames, ignore_index=True)


def build_weekly_volume_drop_signals(weekly_df: pd.DataFrame, start_date: date, end_date: date, config: BacktestConfig) -> pd.DataFrame:
    if weekly_df.empty:
        return pd.DataFrame(columns=["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"])
    frames = []
    trade_dates = sorted(day for day in weekly_df["TradeDate"].dropna().unique() if start_date <= day <= end_date)
    for trade_date in trade_dates:
        selected = compute_weekly_volume_drop_selection(weekly_df, trade_date)
        selected = selected[selected["TradeDate"] == trade_date].copy() if not selected.empty else selected
        if config.limit_per_day is not None and config.limit_per_day > 0:
            selected = selected.head(config.limit_per_day)
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"])
    return pd.concat(frames, ignore_index=True)


def build_weekly_volume_drop_signals_stream(conn, start_date: date, end_date: date, config: BacktestConfig) -> pd.DataFrame:
    symbols = load_symbols(conn, "wkandles", "W", start_date - timedelta(days=70), end_date + timedelta(days=7))
    frames = []
    for batch in iter_batches(symbols, SYMBOL_BATCH_SIZE):
        weekly_df = load_backtest_weekly_for_symbols(conn, batch, start_date, end_date)
        selected = build_weekly_volume_drop_signals(weekly_df, start_date, end_date, config)
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"])
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["TradeDate", "Score", "SCode"], ascending=[True, False, True])
    if config.limit_per_day is not None and config.limit_per_day > 0:
        result = result.groupby("TradeDate", group_keys=False).head(config.limit_per_day)
    return result.reset_index(drop=True)


def build_backtest_signals(conn, daily_df: pd.DataFrame, start_date: date, end_date: date, config: BacktestConfig) -> pd.DataFrame:
    if config.strategy_name == STRATEGY_MA_BULLISH:
        return build_ma_bullish_signals(daily_df, start_date, end_date, config)
    if config.strategy_name == STRATEGY_NEWS_HOT:
        return build_news_hot_signals(conn, daily_df, start_date, end_date, config)
    if config.strategy_name == STRATEGY_WEEKLY_VOLUME_DROP:
        weekly_df = load_backtest_weekly(conn, start_date, end_date)
        return build_weekly_volume_drop_signals(weekly_df, start_date, end_date, config)
    raise ValueError(f"unsupported strategy: {config.strategy_name}")


def run_weekly_volume_drop_backtest(config: BacktestConfig, start_date: date, end_date: date) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    with mysql_connect() as conn:
        selected = build_weekly_volume_drop_signals_stream(conn, start_date, end_date, config)
        symbols = sorted(selected["SCode"].dropna().unique().tolist()) if not selected.empty else []
        daily_df = load_backtest_daily_for_symbols(conn, symbols, start_date, end_date, config.ktype)
    trade_frames = build_trade_day_positions(daily_df)
    evaluated = []
    for row in selected.itertuples(index=False):
        item = evaluate_selection(row, trade_frames)
        if item is not None:
            evaluated.append(item)
    results = pd.DataFrame(evaluated)
    summary = summarize_results(results)
    stock_summary = summarize_results_by_selection(results, start_date, end_date)
    if config.output:
        results.to_csv(config.output, index=False, encoding="utf-8-sig")
    if config.save_db:
        with mysql_connect() as conn:
            saved = save_backtest_summary(conn, stock_summary)
        summary["db_saved"] = saved
    return results, summary, stock_summary


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
            "avg_rise_rate": 0.0,
            "avg_drop_rate": 0.0,
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
        "avg_rise_rate": float(results["RiseRate"].mean(skipna=True)),
        "avg_drop_rate": float(results["WeightedDropRate"].mean(skipna=True)),
    }


def summarize_results_by_stock(results: pd.DataFrame, strategy_name: str, start_date: date, end_date: date) -> pd.DataFrame:
    columns = [
        "SCode",
        "StrategyName",
        "StartDate",
        "EndDate",
        "SampleCount",
        "SuccessRate",
        "AvgRiseRate",
        "FailureRate",
        "AvgDropRate",
        "ExplosiveRate",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for symbol, group in results.groupby("SCode", sort=True):
        sample_count = len(group)
        rows.append(
            {
                "SCode": symbol,
                "StrategyName": strategy_name,
                "StartDate": start_date,
                "EndDate": end_date,
                "SampleCount": sample_count,
                "SuccessRate": float(group["Success"].sum() / sample_count),
                "AvgRiseRate": float(group["RiseRate"].mean(skipna=True)),
                "FailureRate": float(group["Failure"].sum() / sample_count),
                "AvgDropRate": float(group["WeightedDropRate"].mean(skipna=True)),
                "ExplosiveRate": float(group["Explosive"].sum() / sample_count),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_results_by_selection(results: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    columns = [
        "SCode",
        "StrategyName",
        "StartDate",
        "EndDate",
        "SelectionDate",
        "SampleCount",
        "SuccessRate",
        "AvgRiseRate",
        "FailureRate",
        "AvgDropRate",
        "ExplosiveRate",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in results.itertuples(index=False):
        rows.append(
            {
                "SCode": row.SCode,
                "StrategyName": getattr(row, "StrategyName", DEFAULT_STRATEGY_NAME),
                "StartDate": start_date,
                "EndDate": end_date,
                "SelectionDate": row.TradeDate,
                "SampleCount": 1,
                "SuccessRate": 1.0 if row.Success else 0.0,
                "AvgRiseRate": row.RiseRate,
                "FailureRate": 1.0 if row.Failure else 0.0,
                "AvgDropRate": row.WeightedDropRate,
                "ExplosiveRate": 1.0 if row.Explosive else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def save_backtest_summary(
    conn: pymysql.connections.Connection,
    summary_df: pd.DataFrame,
) -> int:
    ensure_backtest_result_table(conn)
    if summary_df.empty:
        return 0
    now = datetime.now()
    rows = []
    for row in summary_df.itertuples(index=False):
        selection_date = getattr(row, "SelectionDate", None)
        rows.append(
            (
                row.SCode,
                row.StrategyName,
                row.StartDate,
                row.EndDate,
                selection_date,
                row.SampleCount,
                none_if_nan(row.SuccessRate),
                none_if_nan(row.AvgRiseRate),
                none_if_nan(row.FailureRate),
                none_if_nan(row.AvgDropRate),
                none_if_nan(row.ExplosiveRate),
                now,
            )
        )
    sql = f"""
        INSERT INTO {BACKTEST_RESULT_TABLE}
            (SCode, StrategyName, StartDate, EndDate, SelectionDate, SampleCount, SuccessRate, AvgRiseRate, FailureRate, AvgDropRate, ExplosiveRate, CreatedOn)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            SelectionDate = VALUES(SelectionDate),
            SampleCount = VALUES(SampleCount),
            SuccessRate = VALUES(SuccessRate),
            AvgRiseRate = VALUES(AvgRiseRate),
            FailureRate = VALUES(FailureRate),
            AvgDropRate = VALUES(AvgDropRate),
            ExplosiveRate = VALUES(ExplosiveRate),
            CreatedOn = VALUES(CreatedOn)
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def run_backtest(config: BacktestConfig) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    start_date = parse_date(config.start_date)
    end_date = parse_date(config.end_date)
    if start_date is None or end_date is None:
        raise ValueError("start-date and end-date are required")
    if start_date > end_date:
        raise ValueError("start-date must be <= end-date")
    if config.strategy_name == STRATEGY_WEEKLY_VOLUME_DROP:
        return run_weekly_volume_drop_backtest(config, start_date, end_date)

    with mysql_connect() as conn:
        daily_df = load_backtest_daily(conn, start_date, end_date, config.ktype)
        selected = build_backtest_signals(conn, daily_df, start_date, end_date, config)
    trade_frames = build_trade_day_positions(daily_df)
    evaluated = []
    for row in selected.itertuples(index=False):
        item = evaluate_selection(row, trade_frames)
        if item is not None:
            evaluated.append(item)
    results = pd.DataFrame(evaluated)
    summary = summarize_results(results)
    stock_summary = summarize_results_by_selection(results, start_date, end_date)
    if config.output:
        results.to_csv(config.output, index=False, encoding="utf-8-sig")
    if config.save_db:
        with mysql_connect() as conn:
            saved = save_backtest_summary(conn, stock_summary)
        summary["db_saved"] = saved
    return results, summary, stock_summary


def print_summary(summary: dict) -> None:
    print(f"total={summary['total']}")
    print(f"success={summary['success_count']} success_rate={summary['success_rate']:.2%}")
    print(f"avg_rise_rate={summary['avg_rise_rate']:.2%}")
    print(f"failure={summary['failure_count']} failure_rate={summary['failure_rate']:.2%}")
    print(f"avg_drop_rate={summary['avg_drop_rate']:.2%}")
    print(f"explosive={summary['explosive_count']} explosive_rate={summary['explosive_rate']:.2%}")
    if "db_saved" in summary:
        print(f"db_saved={summary['db_saved']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest A-share stock selection strategy")
    parser.add_argument("--start-date", required=True, help="Backtest start date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Backtest end date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--min-turnover-amount", type=float, default=0.0, help="Minimum 5-day average Amount for strategy")
    parser.add_argument("--limit-per-day", type=int, default=None, help="Maximum selected stocks per trade date")
    parser.add_argument("--ktype", default=DEFAULT_KTYPE)
    parser.add_argument("--output", help="Optional CSV path for detailed backtest rows")
    parser.add_argument("--strategy-name", choices=STRATEGIES, default=DEFAULT_STRATEGY_NAME, help="Strategy to backtest")
    parser.add_argument("--min-recommendations", type=int, default=3, help="Minimum recommendations for news_hot_v1")
    parser.add_argument("--max-recommendations", type=int, default=5, help="Maximum recommendations for news_hot_v1")
    parser.add_argument("--no-save-db", action="store_true", help="Do not save per-stock backtest summary to MySQL")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, summary, _ = run_backtest(
        BacktestConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            min_turnover_amount=args.min_turnover_amount,
            limit_per_day=args.limit_per_day,
            ktype=args.ktype,
            output=args.output,
            strategy_name=args.strategy_name,
            save_db=not args.no_save_db,
            min_recommendations=args.min_recommendations,
            max_recommendations=args.max_recommendations,
        )
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
