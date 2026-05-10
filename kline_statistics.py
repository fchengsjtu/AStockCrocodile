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
DEFAULT_SYMBOL_BATCH_SIZE = 80
DEFAULT_NEWS_WINDOW_DAYS = 3
STAT_COLUMNS = ["SCode", "SName", "StartRiseDate", "PrevTradeDate", "SelectionDate", "GainRate", "StatType"]
MESSAGE_DRIVEN_KEYWORDS = (
    "公告",
    "利好",
    "重大",
    "重组",
    "并购",
    "收购",
    "借壳",
    "中标",
    "合同",
    "订单",
    "业绩",
    "预增",
    "扭亏",
    "涨价",
    "政策",
    "获批",
    "批复",
    "新药",
    "临床",
    "复牌",
    "停牌",
    "回购",
    "增持",
    "股权",
    "实控人",
    "注入",
)


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
    batch_size: int
    news_filter: bool
    news_window_days: int


def ensure_kline_statistics_table(conn: pymysql.connections.Connection) -> None:
    sql = f"""
        CREATE TABLE IF NOT EXISTS {KLINE_STATISTICS_TABLE} (
            Id BIGINT NOT NULL AUTO_INCREMENT,
            SCode VARCHAR(10) NOT NULL,
            SName VARCHAR(64) NULL,
            StartRiseDate DATE NOT NULL,
            PrevTradeDate DATE NOT NULL,
            SelectionDate DATE NULL,
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
        cur.execute(f"SHOW COLUMNS FROM {KLINE_STATISTICS_TABLE} LIKE 'SelectionDate'")
        if cur.fetchone() is None:
            cur.execute(f"ALTER TABLE {KLINE_STATISTICS_TABLE} ADD COLUMN SelectionDate DATE NULL AFTER PrevTradeDate")
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


def normalize_daily_kline_rows(rows) -> pd.DataFrame:
    columns = ["SCode", "SName", "TradeDate", "Close"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df


def iter_batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def load_symbols(conn: pymysql.connections.Connection, start_date: date, end_date: date, ktype: str) -> list[str]:
    sql = """
        SELECT DISTINCT SCode
        FROM dkandles
        WHERE KType = %s AND KTime >= %s AND KTime < %s
        ORDER BY SCode
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ktype, start_date, end_date + timedelta(days=1)))
        rows = cur.fetchall()
    return [row[0] for row in rows]


def load_daily_kline_for_symbols(
    conn: pymysql.connections.Connection,
    symbols: list[str],
    start_date: date,
    end_date: date,
    ktype: str,
    forward_days: int,
) -> pd.DataFrame:
    if not symbols:
        return normalize_daily_kline_rows([])
    load_start = start_date - timedelta(days=10)
    load_end = end_date + timedelta(days=max(10, forward_days * 5))
    placeholders = ",".join(["%s"] * len(symbols))
    sql = f"""
        SELECT dk.SCode, si.SName, DATE(dk.KTime) AS TradeDate, dk.Close
        FROM dkandles dk
        LEFT JOIN stockinfo si ON si.SCode = dk.SCode
        WHERE dk.KType = %s
          AND dk.SCode IN ({placeholders})
          AND dk.KTime >= %s AND dk.KTime < %s
        ORDER BY dk.SCode, dk.KTime
    """
    params = [ktype, *symbols, load_start, load_end + timedelta(days=1)]
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return normalize_daily_kline_rows(rows)


def find_short_term_surges(
    daily_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    forward_days: int = 3,
    threshold: float = 0.20,
    stat_type: str = SHORT_TERM_SURGE_TYPE,
) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(columns=STAT_COLUMNS)

    rows = []
    for symbol, group in daily_df.groupby("SCode", sort=False):
        frame = group.sort_values("TradeDate").reset_index(drop=True)
        for pos, item in frame.iterrows():
            trade_date = item["TradeDate"]
            if trade_date < start_date or trade_date > end_date:
                continue
            if pos < 2 or pos + forward_days >= len(frame):
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
                        "SelectionDate": frame.iloc[pos - 2]["TradeDate"],
                        "GainRate": gain_rate,
                        "StatType": stat_type,
                    }
                )
    return pd.DataFrame(rows, columns=STAT_COLUMNS)


def news_table_exists(conn: pymysql.connections.Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'news'")
        return cur.fetchone() is not None


def has_message_driven_news(stock_code: str, stock_name: str | None, title: str | None, summary: str | None) -> bool:
    text = f"{title or ''} {summary or ''}"
    compact_code = str(stock_code).strip()
    name = str(stock_name or "").strip()
    if not text:
        return False
    mentioned = (name and name in text) or (compact_code and compact_code in text)
    if not mentioned:
        return False
    return any(keyword in text for keyword in MESSAGE_DRIVEN_KEYWORDS)


def load_news_for_stats(conn: pymysql.connections.Connection, stats: pd.DataFrame, window_days: int) -> pd.DataFrame:
    columns = ["PublishDate", "Title", "Summary"]
    if stats.empty:
        return pd.DataFrame(columns=columns)
    if not news_table_exists(conn):
        return pd.DataFrame(columns=columns)
    date_column = "SelectionDate" if "SelectionDate" in stats.columns else "StartRiseDate"
    min_date = min(stats[date_column]) - timedelta(days=window_days)
    max_date = max(stats[date_column]) + timedelta(days=window_days)
    sql = """
        SELECT DATE(PublishTime) AS PublishDate, Title, Summary
        FROM news
        WHERE PublishTime >= %s AND PublishTime < %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (min_date, max_date + timedelta(days=1)))
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["PublishDate"] = pd.to_datetime(df["PublishDate"]).dt.date
    return df


def filter_message_driven_surges(
    stats: pd.DataFrame,
    news_df: pd.DataFrame,
    window_days: int,
) -> tuple[pd.DataFrame, int]:
    if stats.empty or news_df.empty:
        return stats, 0
    keep_rows = []
    excluded = 0
    for row in stats.itertuples(index=False):
        reference_date = getattr(row, "SelectionDate", row.StartRiseDate)
        nearby = news_df[
            (news_df["PublishDate"] >= reference_date - timedelta(days=window_days))
            & (news_df["PublishDate"] <= reference_date + timedelta(days=window_days))
        ]
        message_driven = False
        for news in nearby.itertuples(index=False):
            if has_message_driven_news(row.SCode, row.SName, news.Title, news.Summary):
                message_driven = True
                break
        if message_driven:
            excluded += 1
        else:
            keep_rows.append(row._asdict())
    if not keep_rows:
        return pd.DataFrame(columns=stats.columns), excluded
    return pd.DataFrame(keep_rows, columns=stats.columns), excluded


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
                getattr(row, "SelectionDate", None),
                none_if_nan(row.GainRate),
                row.StatType,
                now,
            )
        )
    sql = f"""
        INSERT INTO {KLINE_STATISTICS_TABLE}
            (SCode, SName, StartRiseDate, PrevTradeDate, SelectionDate, GainRate, StatType, CreatedOn)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            SName = VALUES(SName),
            PrevTradeDate = VALUES(PrevTradeDate),
            SelectionDate = VALUES(SelectionDate),
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

    frames = []
    matched = 0
    excluded_news = 0
    saved = 0
    wrote_header = False
    with mysql_connect() as conn:
        if config.save_db:
            ensure_kline_statistics_table(conn)
        symbols = load_symbols(conn, start_date, end_date, config.ktype)
        batches = list(iter_batches(symbols, config.batch_size))
        print(
            f"start kline statistics stat_type={config.stat_type} "
            f"symbols={len(symbols)} batches={len(batches)} batch_size={config.batch_size} "
            f"news_filter={config.news_filter} news_window_days={config.news_window_days}",
            flush=True,
        )
        for batch_index, batch in enumerate(batches, start=1):
            daily_df = load_daily_kline_for_symbols(conn, batch, start_date, end_date, config.ktype, config.forward_days)
            stats = find_short_term_surges(
                daily_df=daily_df,
                start_date=start_date,
                end_date=end_date,
                forward_days=config.forward_days,
                threshold=config.threshold,
                stat_type=config.stat_type,
            )
            batch_matched = len(stats)
            batch_excluded = 0
            matched += batch_matched
            if config.news_filter and not stats.empty:
                news_df = load_news_for_stats(conn, stats, config.news_window_days)
                stats, batch_excluded = filter_message_driven_surges(stats, news_df, config.news_window_days)
                excluded_news += batch_excluded
            if config.output and not stats.empty:
                stats.to_csv(
                    config.output,
                    mode="w" if not wrote_header else "a",
                    header=not wrote_header,
                    index=False,
                    encoding="utf-8-sig",
                )
                wrote_header = True
            if config.save_db and not stats.empty:
                saved += save_kline_statistics(conn, stats)
            if not stats.empty:
                frames.append(stats)
            print(
                f"batch {batch_index}/{len(batches)} symbols={len(batch)} "
                f"rows={len(daily_df)} matched={batch_matched} "
                f"excluded_news={batch_excluded} kept={len(stats)} "
                f"total_matched={matched} total_excluded_news={excluded_news} total_saved={saved}",
                flush=True,
            )
    if config.output and not wrote_header:
        pd.DataFrame(columns=STAT_COLUMNS).to_csv(
            config.output,
            index=False,
            encoding="utf-8-sig",
        )
    stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STAT_COLUMNS)
    print(f"stat_type={config.stat_type} matched={matched} excluded_news={excluded_news} kept={len(stats)} saved={saved}")
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
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SYMBOL_BATCH_SIZE, help="Number of stock symbols loaded per batch")
    parser.add_argument("--news-window-days", type=int, default=DEFAULT_NEWS_WINDOW_DAYS, help="Days before and after SelectionDate to scan news")
    parser.add_argument("--no-news-filter", action="store_true", help="Do not exclude message-driven surges using the news table")
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
            batch_size=max(1, args.batch_size),
            news_filter=not args.no_news_filter,
            news_window_days=max(0, args.news_window_days),
        )
    )


if __name__ == "__main__":
    main()
