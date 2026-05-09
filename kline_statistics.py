from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import pymysql

from a_share_crawler import DEFAULT_KTYPE, mysql_connect, none_if_nan
from stock_selector import parse_date

KLINE_STATISTICS_TABLE = "klinestatistics"
SHORT_TERM_SURGE_TYPE = "short_term_surge_3d_20pct"


@dataclass(frozen=True)
class KlineStatisticsConfig:
    start_date: str
    end_date: str
    ktype: str
    stat_type: str
    forward_days: int
    threshold: float
    output: str | None
    save_db: bool


def ensure_kline_statistics_table(conn: pymysql.connections.Connection) -> None:
    sql = f"""
        CREATE TABLE IF NOT EXISTS {KLINE_STATISTICS_TABLE} (
            Id BIGINT NOT NULL AUTO_INCREMENT,
            SCode VARCHAR(10) NOT NULL,
            SName VARCHAR(64) NULL,
            StartRiseDate DATE NOT NULL,
            PrevTradeDate DATE NOT NULL,
            GainRate DECIMAL(18,6) NOT NULL,
            StatType VARCHAR(64) NOT NULL,
            CreatedOn DATETIME NOT NULL,
            PRIMARY KEY (Id),
            UNIQUE KEY ux_klinestatistics_code_type_start (SCode, StatType, StartRiseDate),
            KEY idx_klinestatistics_type_date (StatType, StartRiseDate)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def load_daily_kline(conn: pymysql.connections.Connection, start_date: date, end_date: date, ktype: str, forward_days: int) -> pd.DataFrame:
    load_start = start_date - timedelta(days=10)
    load_end = end_date + timedelta(days=max(10, forward_days * 5))
    sql = """
        SELECT dk.SCode, si.SName, DATE(dk.KTime) AS TradeDate, dk.Close
        FROM dkandles dk
        LEFT JOIN stockinfo si ON si.SCode = dk.SCode
        WHERE dk.KType = %s AND dk.KTime >= %s AND dk.KTime < %s
        ORDER BY dk.SCode, dk.KTime
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ktype, load_start, load_end + timedelta(days=1)))
        rows = cur.fetchall()
    columns = ["SCode", "SName", "TradeDate", "Close"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df


def find_short_term_surges(
    daily_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    forward_days: int = 3,
    threshold: float = 0.20,
    stat_type: str = SHORT_TERM_SURGE_TYPE,
) -> pd.DataFrame:
    columns = ["SCode", "SName", "StartRiseDate", "PrevTradeDate", "GainRate", "StatType"]
    if daily_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for symbol, group in daily_df.groupby("SCode", sort=False):
        frame = group.sort_values("TradeDate").reset_index(drop=True)
        for pos, item in frame.iterrows():
            trade_date = item["TradeDate"]
            if trade_date < start_date or trade_date > end_date:
                continue
            if pos == 0 or pos + forward_days >= len(frame):
                continue
            close_price = item["Close"]
            future_close = frame.iloc[pos + forward_days]["Close"]
            if pd.isna(close_price) or pd.isna(future_close) or close_price <= 0:
                continue
            gain_rate = float(future_close / close_price - 1)
            if gain_rate >= threshold:
                rows.append(
                    {
                        "SCode": symbol,
                        "SName": item["SName"],
                        "StartRiseDate": trade_date,
                        "PrevTradeDate": frame.iloc[pos - 1]["TradeDate"],
                        "GainRate": gain_rate,
                        "StatType": stat_type,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def save_kline_statistics(conn: pymysql.connections.Connection, stats: pd.DataFrame) -> int:
    ensure_kline_statistics_table(conn)
    if stats.empty:
        return 0
    now = datetime.now()
    rows = []
    for row in stats.itertuples(index=False):
        rows.append(
            (
                row.SCode,
                none_if_nan(row.SName),
                row.StartRiseDate,
                row.PrevTradeDate,
                none_if_nan(row.GainRate),
                row.StatType,
                now,
            )
        )
    sql = f"""
        INSERT INTO {KLINE_STATISTICS_TABLE}
            (SCode, SName, StartRiseDate, PrevTradeDate, GainRate, StatType, CreatedOn)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            SName = VALUES(SName),
            PrevTradeDate = VALUES(PrevTradeDate),
            GainRate = VALUES(GainRate),
            CreatedOn = VALUES(CreatedOn)
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def run_statistics(config: KlineStatisticsConfig) -> pd.DataFrame:
    start_date = parse_date(config.start_date)
    end_date = parse_date(config.end_date)
    if start_date is None or end_date is None:
        raise ValueError("start-date and end-date are required")
    if start_date > end_date:
        raise ValueError("start-date must be <= end-date")

    with mysql_connect() as conn:
        daily_df = load_daily_kline(conn, start_date, end_date, config.ktype, config.forward_days)
    stats = find_short_term_surges(
        daily_df=daily_df,
        start_date=start_date,
        end_date=end_date,
        forward_days=config.forward_days,
        threshold=config.threshold,
        stat_type=config.stat_type,
    )
    if config.output:
        stats.to_csv(config.output, index=False, encoding="utf-8-sig")
    saved = 0
    if config.save_db:
        with mysql_connect() as conn:
            saved = save_kline_statistics(conn, stats)
    print(f"stat_type={config.stat_type} matched={len(stats)} saved={saved}")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute statistics from daily K-line data")
    parser.add_argument("--start-date", required=True, help="Statistics start date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Statistics end date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--ktype", default=DEFAULT_KTYPE)
    parser.add_argument("--stat-type", default=SHORT_TERM_SURGE_TYPE, help="Statistic type stored in database")
    parser.add_argument("--forward-days", type=int, default=3, help="Trading days after the start-rise date")
    parser.add_argument("--threshold", type=float, default=0.20, help="Minimum gain rate, 0.20 means 20 percent")
    parser.add_argument("--output", help="Optional CSV path for matched rows")
    parser.add_argument("--no-save-db", action="store_true", help="Do not save statistics to MySQL")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_statistics(
        KlineStatisticsConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            ktype=args.ktype,
            stat_type=args.stat_type,
            forward_days=args.forward_days,
            threshold=args.threshold,
            output=args.output,
            save_db=not args.no_save_db,
        )
    )


if __name__ == "__main__":
    main()
