from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pymysql

from llm_finetune.common import mysql_connect
from stock_selector import STRATEGY_NEWS_HOT, STRATEGY_WEEKLY_VOLUME_DROP

from .blackbox import build_blackbox_signals
from .common import PortfolioBacktestConfig, filter_selection_candidates, is_blackbox_strategy


DAILY_TABLE = "portfolio_backtest_daily"
HOLDING_TABLE = "portfolio_backtest_holdings"
TRADE_TABLE = "portfolio_backtest_trades"


def none_if_nan(value):
    if pd.isna(value):
        return None
    return value


def iter_batches(items: list[str], batch_size: int):
    for index in range(0, len(items), max(1, batch_size)):
        yield items[index : index + max(1, batch_size)]


def normalize_daily_frame(rows) -> pd.DataFrame:
    columns = ["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume", "MA5", "MA8", "MA13", "MA34", "MA55"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    for column in columns[3:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_daily_for_symbols(
    conn: pymysql.connections.Connection,
    symbols: list[str],
    start_date: date,
    end_date: date,
    ktype: str,
    batch_size: int,
) -> pd.DataFrame:
    if not symbols:
        return normalize_daily_frame([])
    frames = []
    for batch in iter_batches(symbols, batch_size):
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
        params = [ktype, *batch, start_date, end_date + timedelta(days=1)]
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        frame = normalize_daily_frame(rows)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return normalize_daily_frame([])
    return pd.concat(frames, ignore_index=True)


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


def ensure_portfolio_tables(conn: pymysql.connections.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {DAILY_TABLE} (
                Id BIGINT NOT NULL AUTO_INCREMENT,
                BacktestName VARCHAR(128) NOT NULL,
                TradeDate DATE NOT NULL,
                StrategyName VARCHAR(64) NOT NULL,
                TradeRuleName VARCHAR(128) NOT NULL DEFAULT 'stop_loss_3pct_take_profit_10_20_hold_3d',
                SelectionRule VARCHAR(512) NULL,
                ExitRule VARCHAR(512) NULL,
                TotalMarketValue DECIMAL(24,6) NOT NULL,
                HoldingMarketValue DECIMAL(24,6) NOT NULL,
                CashAmount DECIMAL(24,6) NOT NULL,
                ActualProfit DECIMAL(24,6) NOT NULL DEFAULT 0,
                DailyBuyAmount DECIMAL(24,6) NOT NULL,
                DailySellAmount DECIMAL(24,6) NOT NULL,
                TradingFee DECIMAL(24,6) NOT NULL,
                PositionCount INT NOT NULL,
                CreatedOn DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (Id),
                UNIQUE KEY ux_portfolio_daily (BacktestName, StrategyName, TradeRuleName, TradeDate),
                KEY idx_portfolio_daily_date (TradeDate)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {HOLDING_TABLE} (
                Id BIGINT NOT NULL AUTO_INCREMENT,
                BacktestName VARCHAR(128) NOT NULL,
                TradeDate DATE NOT NULL,
                SCode VARCHAR(10) NOT NULL,
                SName VARCHAR(64) NULL,
                StrategyName VARCHAR(64) NOT NULL,
                TradeRuleName VARCHAR(128) NOT NULL DEFAULT 'stop_loss_3pct_take_profit_10_20_hold_3d',
                SelectionDate DATE NOT NULL,
                BuyDate DATE NOT NULL,
                Shares INT NOT NULL,
                CostPrice DECIMAL(18,6) NOT NULL,
                ClosePrice DECIMAL(18,6) NOT NULL,
                MarketValue DECIMAL(24,6) NOT NULL,
                UnrealizedPnl DECIMAL(24,6) NOT NULL,
                SelectionRule VARCHAR(512) NULL,
                ExitRule VARCHAR(512) NULL,
                CreatedOn DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (Id),
                UNIQUE KEY ux_portfolio_holding (BacktestName, StrategyName, TradeRuleName, TradeDate, SCode, SelectionDate, BuyDate),
                KEY idx_portfolio_holding_code (SCode, TradeDate)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TRADE_TABLE} (
                Id BIGINT NOT NULL AUTO_INCREMENT,
                BacktestName VARCHAR(128) NOT NULL,
                TradeDate DATE NOT NULL,
                SCode VARCHAR(10) NOT NULL,
                SName VARCHAR(64) NULL,
                StrategyName VARCHAR(64) NOT NULL,
                TradeRuleName VARCHAR(128) NOT NULL DEFAULT 'stop_loss_3pct_take_profit_10_20_hold_3d',
                SelectionDate DATE NOT NULL,
                Side VARCHAR(8) NOT NULL,
                Shares INT NOT NULL,
                Price DECIMAL(18,6) NOT NULL,
                GrossAmount DECIMAL(24,6) NOT NULL,
                Fee DECIMAL(24,6) NOT NULL,
                NetAmount DECIMAL(24,6) NOT NULL,
                Reason VARCHAR(128) NOT NULL,
                SelectionRule VARCHAR(512) NULL,
                ExitRule VARCHAR(512) NULL,
                CreatedOn DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (Id),
                KEY idx_portfolio_trade_date (BacktestName, TradeDate),
                KEY idx_portfolio_trade_code (SCode, TradeDate)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        ensure_trade_rule_column(cur, DAILY_TABLE, "StrategyName")
        ensure_trade_rule_column(cur, HOLDING_TABLE, "StrategyName")
        ensure_trade_rule_column(cur, TRADE_TABLE, "StrategyName")
        ensure_actual_profit_column(cur)
        ensure_unique_index(
            cur,
            DAILY_TABLE,
            "ux_portfolio_daily",
            "BacktestName, StrategyName, TradeRuleName, TradeDate",
        )
        ensure_unique_index(
            cur,
            HOLDING_TABLE,
            "ux_portfolio_holding",
            "BacktestName, StrategyName, TradeRuleName, TradeDate, SCode, SelectionDate, BuyDate",
        )
    conn.commit()


def ensure_trade_rule_column(cur, table: str, after_column: str) -> None:
    cur.execute(f"SHOW COLUMNS FROM {table} LIKE 'TradeRuleName'")
    if cur.fetchone() is None:
        cur.execute(
            f"ALTER TABLE {table} "
            "ADD COLUMN TradeRuleName VARCHAR(128) NOT NULL DEFAULT 'stop_loss_3pct_take_profit_10_20_hold_3d' "
            f"AFTER {after_column}"
        )


def ensure_actual_profit_column(cur) -> None:
    cur.execute(f"SHOW COLUMNS FROM {DAILY_TABLE} LIKE 'ActualProfit'")
    if cur.fetchone() is None:
        cur.execute(
            f"ALTER TABLE {DAILY_TABLE} "
            "ADD COLUMN ActualProfit DECIMAL(24,6) NOT NULL DEFAULT 0 "
            "AFTER CashAmount"
        )


def ensure_unique_index(cur, table: str, index_name: str, index_columns: str) -> None:
    cur.execute(f"SHOW INDEX FROM {table} WHERE Key_name = %s", (index_name,))
    rows = cur.fetchall()
    existing_columns = ",".join(row[4] for row in rows) if rows else ""
    desired_columns = ",".join(item.strip() for item in index_columns.split(","))
    if existing_columns == desired_columns:
        return
    if rows:
        cur.execute(f"ALTER TABLE {table} DROP INDEX {index_name}")
    cur.execute(f"ALTER TABLE {table} ADD UNIQUE KEY {index_name} ({index_columns})")


def clear_backtest_rows(conn: pymysql.connections.Connection, backtest_name: str, strategy_name: str, trade_rule_name: str | None = None) -> None:
    with conn.cursor() as cur:
        for table in (DAILY_TABLE, HOLDING_TABLE, TRADE_TABLE):
            if trade_rule_name is None:
                cur.execute(f"DELETE FROM {table} WHERE BacktestName = %s AND StrategyName = %s", (backtest_name, strategy_name))
            else:
                cur.execute(
                    f"DELETE FROM {table} WHERE BacktestName = %s AND StrategyName = %s AND TradeRuleName = %s",
                    (backtest_name, strategy_name, trade_rule_name),
                )
    conn.commit()


def save_dataframe(conn: pymysql.connections.Connection, table: str, df: pd.DataFrame, columns: list[str]) -> int:
    if df.empty:
        return 0
    placeholders = ",".join(["%s"] * len(columns))
    column_sql = ", ".join(columns)
    sql = f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})"
    rows = [tuple(none_if_nan(row[column]) for column in columns) for _, row in df.iterrows()]
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def save_results(conn: pymysql.connections.Connection, daily: pd.DataFrame, holdings: pd.DataFrame, trades: pd.DataFrame) -> dict[str, int]:
    ensure_portfolio_tables(conn)
    counts = {
        "daily": save_dataframe(
            conn,
            DAILY_TABLE,
            daily,
            [
                "BacktestName",
                "TradeDate",
                "StrategyName",
                "TradeRuleName",
                "SelectionRule",
                "ExitRule",
                "TotalMarketValue",
                "HoldingMarketValue",
                "CashAmount",
                "ActualProfit",
                "DailyBuyAmount",
                "DailySellAmount",
                "TradingFee",
                "PositionCount",
            ],
        ),
        "holdings": save_dataframe(
            conn,
            HOLDING_TABLE,
            holdings,
            [
                "BacktestName",
                "TradeDate",
                "SCode",
                "SName",
                "StrategyName",
                "TradeRuleName",
                "SelectionDate",
                "BuyDate",
                "Shares",
                "CostPrice",
                "ClosePrice",
                "MarketValue",
                "UnrealizedPnl",
                "SelectionRule",
                "ExitRule",
            ],
        ),
        "trades": save_dataframe(
            conn,
            TRADE_TABLE,
            trades,
            [
                "BacktestName",
                "TradeDate",
                "SCode",
                "SName",
                "StrategyName",
                "TradeRuleName",
                "SelectionDate",
                "Side",
                "Shares",
                "Price",
                "GrossAmount",
                "Fee",
                "NetAmount",
                "Reason",
                "SelectionRule",
                "ExitRule",
            ],
        ),
    }
    return counts


def load_strategy_signals(conn: pymysql.connections.Connection, config: PortfolioBacktestConfig) -> pd.DataFrame:
    if is_blackbox_strategy(config.strategy_name):
        return build_blackbox_signals(conn, config)
    from backtest_strategy import BacktestConfig, build_signals_stream

    backtest_config = BacktestConfig(
        start_date=config.start_date.strftime("%Y%m%d"),
        end_date=config.end_date.strftime("%Y%m%d"),
        min_turnover_amount=config.min_turnover_amount,
        limit_per_day=None,
        ktype=config.ktype,
        output=None,
        strategy_name=config.strategy_name,
        save_db=False,
        min_recommendations=config.min_recommendations,
        max_recommendations=config.max_recommendations,
        batch_size=config.batch_size,
    )
    signals = build_signals_stream(conn, config.start_date, config.end_date, backtest_config)
    trade_dates = load_market_trade_dates(conn, config.start_date, config.end_date, config.ktype)
    return filter_selection_candidates(
        signals,
        trade_dates,
        cooldown_trading_days=config.selection_cooldown_trading_days,
        limit_per_day=config.limit_per_day,
    )


def load_market_trade_dates(conn, start_date: date, end_date: date, ktype: str = "D") -> list[date]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT DATE(KTime)
            FROM dkandles
            WHERE KType = %s AND KTime >= %s AND KTime < %s
            ORDER BY DATE(KTime)
            """,
            (ktype, start_date, end_date + timedelta(days=1)),
        )
        rows = cur.fetchall()
    return [row[0] for row in rows]


def load_daily_for_simulation(conn: pymysql.connections.Connection, signals: pd.DataFrame, config: PortfolioBacktestConfig) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    symbols = sorted(signals["SCode"].dropna().astype(str).unique().tolist())
    load_end = config.end_date + timedelta(days=15)
    load_start = config.start_date - timedelta(days=180)
    load_end = load_end + timedelta(days=32)
    return load_daily_for_symbols(conn, symbols, load_start, load_end, config.ktype, config.batch_size)


def available_symbols(conn: pymysql.connections.Connection, config: PortfolioBacktestConfig) -> list[str]:
    if config.strategy_name == STRATEGY_WEEKLY_VOLUME_DROP:
        return load_symbols(conn, "wkandles", "W", config.start_date - timedelta(days=70), config.end_date + timedelta(days=7))
    return load_symbols(conn, "dkandles", config.ktype, config.start_date, config.end_date + timedelta(days=15))


def connect():
    return mysql_connect()
