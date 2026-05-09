from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import pymysql

from a_share_crawler import DEFAULT_KTYPE, mysql_connect, none_if_nan

SELECTION_TABLE = "stockselection"
DEFAULT_LOOKBACK_DAYS = 140


@dataclass(frozen=True)
class SelectionConfig:
    trade_date: str | None
    min_turnover_amount: float
    limit: int | None
    ktype: str


def parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid date: {value}")
    return parsed.date()


def ensure_selection_table(conn: pymysql.connections.Connection) -> None:
    sql = f"""
        CREATE TABLE IF NOT EXISTS {SELECTION_TABLE} (
            Id BIGINT NOT NULL AUTO_INCREMENT,
            TradeDate DATE NOT NULL,
            SCode VARCHAR(10) NOT NULL,
            SName VARCHAR(64) NULL,
            ClosePrice DECIMAL(18,6) NOT NULL,
            Score DECIMAL(18,6) NULL,
            Reason VARCHAR(255) NULL,
            CreatedOn DATETIME NOT NULL,
            PRIMARY KEY (Id),
            UNIQUE KEY ux_stockselection_date_code (TradeDate, SCode),
            KEY idx_stockselection_code (SCode)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def latest_trade_date(conn: pymysql.connections.Connection, ktype: str = DEFAULT_KTYPE) -> date:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(DATE(KTime)) FROM dkandles WHERE KType = %s", (ktype,))
        value = cur.fetchone()[0]
    if value is None:
        raise RuntimeError("dkandles has no daily K-line data")
    return value if isinstance(value, date) else pd.to_datetime(value).date()


def load_daily_window(conn: pymysql.connections.Connection, trade_date: date, lookback_days: int, ktype: str) -> pd.DataFrame:
    start_date = trade_date - timedelta(days=lookback_days * 2)
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
        cur.execute(sql, (ktype, start_date, trade_date + timedelta(days=1)))
        rows = cur.fetchall()
    columns = ["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume", "MA5", "MA8", "MA13", "MA34", "MA55"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    for column in columns[3:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def compute_strategy_frame(daily_df: pd.DataFrame, min_turnover_amount: float = 0.0) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    frames = []
    for _, group in daily_df.groupby("SCode", sort=False):
        item = group.sort_values("TradeDate").copy()
        item["PrevClose"] = item["Close"].shift(1)
        item["AvgAmount5"] = item["Amount"].rolling(5, min_periods=5).mean()
        item["PctChange"] = item["Close"] / item["PrevClose"] - 1
        frames.append(item)
    df = pd.concat(frames, ignore_index=True)

    ma_ready = df[["MA5", "MA8", "MA13", "MA34", "MA55"]].notna().all(axis=1)
    ma_bullish = (df["MA5"] > df["MA8"]) & (df["MA8"] > df["MA13"]) & (df["MA13"] > df["MA34"]) & (df["MA34"] > df["MA55"])
    price_confirm = (df["Close"] > df["MA5"]) & (df["Close"] > df["Open"])
    liquidity = df["AvgAmount5"].fillna(0) >= min_turnover_amount
    not_extreme = df["PctChange"].fillna(0) < 0.095
    df["Selected"] = ma_ready & ma_bullish & price_confirm & liquidity & not_extreme
    df["Score"] = (df["Close"] / df["MA55"] - 1).where(df["MA55"] > 0)
    df["Reason"] = "MA5>MA8>MA13>MA34>MA55; close>MA5; bullish candle"
    return df


def select_stocks_for_date(
    conn: pymysql.connections.Connection,
    trade_date: date,
    min_turnover_amount: float = 0.0,
    limit: int | None = None,
    ktype: str = DEFAULT_KTYPE,
) -> pd.DataFrame:
    daily_df = load_daily_window(conn, trade_date, DEFAULT_LOOKBACK_DAYS, ktype)
    strategy_df = compute_strategy_frame(daily_df, min_turnover_amount=min_turnover_amount)
    if strategy_df.empty:
        return pd.DataFrame()
    selected = strategy_df[(strategy_df["TradeDate"] == trade_date) & strategy_df["Selected"]].copy()
    selected = selected.sort_values(["Score", "Amount"], ascending=[False, False])
    if limit is not None and limit > 0:
        selected = selected.head(limit)
    return selected[["TradeDate", "SCode", "SName", "Close", "Score", "Reason"]]


def save_selections(conn: pymysql.connections.Connection, selected: pd.DataFrame) -> int:
    ensure_selection_table(conn)
    if selected.empty:
        return 0
    now = datetime.now()
    rows = []
    for row in selected.itertuples(index=False):
        rows.append((row.TradeDate, row.SCode, none_if_nan(row.SName), none_if_nan(row.Close), none_if_nan(row.Score), none_if_nan(row.Reason), now))
    sql = f"""
        INSERT INTO {SELECTION_TABLE}
            (TradeDate, SCode, SName, ClosePrice, Score, Reason, CreatedOn)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            SName = VALUES(SName),
            ClosePrice = VALUES(ClosePrice),
            Score = VALUES(Score),
            Reason = VALUES(Reason),
            CreatedOn = VALUES(CreatedOn)
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def run_selection(config: SelectionConfig) -> pd.DataFrame:
    with mysql_connect() as conn:
        trade_date = parse_date(config.trade_date) or latest_trade_date(conn, config.ktype)
        selected = select_stocks_for_date(
            conn=conn,
            trade_date=trade_date,
            min_turnover_amount=config.min_turnover_amount,
            limit=config.limit,
            ktype=config.ktype,
        )
        saved = save_selections(conn, selected)
    print(f"trade_date={trade_date} selected={len(selected)} saved={saved}")
    if not selected.empty:
        print(selected.to_string(index=False))
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run A-share stock selection strategy")
    parser.add_argument("--date", dest="trade_date", help="Selection date, YYYYMMDD or YYYY-MM-DD; default latest trading day in dkandles")
    parser.add_argument("--min-turnover-amount", type=float, default=0.0, help="Minimum 5-day average Amount; Amount follows dkandles unit")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of selected stocks to save")
    parser.add_argument("--ktype", default=DEFAULT_KTYPE)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_selection(
        SelectionConfig(
            trade_date=args.trade_date,
            min_turnover_amount=args.min_turnover_amount,
            limit=args.limit,
            ktype=args.ktype,
        )
    )


if __name__ == "__main__":
    main()
