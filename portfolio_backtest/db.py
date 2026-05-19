from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pymysql

from a_share_crawler import mysql_connect, none_if_nan
from backtest_strategy import BacktestConfig, build_signals_stream, load_backtest_daily_for_symbols, load_symbols
from stock_selector import STRATEGY_NEWS_HOT, STRATEGY_WEEKLY_VOLUME_DROP

from .blackbox import build_blackbox_signals
from .common import PortfolioBacktestConfig, is_blackbox_strategy


DAILY_TABLE = "portfolio_backtest_daily"
HOLDING_TABLE = "portfolio_backtest_holdings"
TRADE_TABLE = "portfolio_backtest_trades"


def ensure_portfolio_tables(conn: pymysql.connections.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {DAILY_TABLE} (
                Id BIGINT NOT NULL AUTO_INCREMENT,
                BacktestName VARCHAR(128) NOT NULL,
                TradeDate DATE NOT NULL,
                StrategyName VARCHAR(64) NOT NULL,
                SelectionRule VARCHAR(512) NULL,
                ExitRule VARCHAR(512) NULL,
                TotalMarketValue DECIMAL(24,6) NOT NULL,
                HoldingMarketValue DECIMAL(24,6) NOT NULL,
                CashAmount DECIMAL(24,6) NOT NULL,
                DailyBuyAmount DECIMAL(24,6) NOT NULL,
                DailySellAmount DECIMAL(24,6) NOT NULL,
                TradingFee DECIMAL(24,6) NOT NULL,
                PositionCount INT NOT NULL,
                CreatedOn DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (Id),
                UNIQUE KEY ux_portfolio_daily (BacktestName, StrategyName, TradeDate),
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
                UNIQUE KEY ux_portfolio_holding (BacktestName, StrategyName, TradeDate, SCode, SelectionDate, BuyDate),
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
    conn.commit()


def clear_backtest_rows(conn: pymysql.connections.Connection, backtest_name: str, strategy_name: str) -> None:
    with conn.cursor() as cur:
        for table in (DAILY_TABLE, HOLDING_TABLE, TRADE_TABLE):
            cur.execute(f"DELETE FROM {table} WHERE BacktestName = %s AND StrategyName = %s", (backtest_name, strategy_name))
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
                "SelectionRule",
                "ExitRule",
                "TotalMarketValue",
                "HoldingMarketValue",
                "CashAmount",
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
    backtest_config = BacktestConfig(
        start_date=config.start_date.strftime("%Y%m%d"),
        end_date=config.end_date.strftime("%Y%m%d"),
        min_turnover_amount=config.min_turnover_amount,
        limit_per_day=config.limit_per_day,
        ktype=config.ktype,
        output=None,
        strategy_name=config.strategy_name,
        save_db=False,
        min_recommendations=config.min_recommendations,
        max_recommendations=config.max_recommendations,
        batch_size=config.batch_size,
    )
    return build_signals_stream(conn, config.start_date, config.end_date, backtest_config)


def load_daily_for_simulation(conn: pymysql.connections.Connection, signals: pd.DataFrame, config: PortfolioBacktestConfig) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    symbols = sorted(signals["SCode"].dropna().astype(str).unique().tolist())
    load_end = config.end_date + timedelta(days=15)
    return load_backtest_daily_for_symbols(conn, symbols, config.start_date, load_end, config.ktype, config.batch_size)


def available_symbols(conn: pymysql.connections.Connection, config: PortfolioBacktestConfig) -> list[str]:
    if config.strategy_name == STRATEGY_WEEKLY_VOLUME_DROP:
        return load_symbols(conn, "wkandles", "W", config.start_date - timedelta(days=70), config.end_date + timedelta(days=7))
    return load_symbols(conn, "dkandles", config.ktype, config.start_date, config.end_date + timedelta(days=15))


def connect():
    return mysql_connect()
