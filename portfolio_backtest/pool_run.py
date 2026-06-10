from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .common import (
    DEFAULT_BACKTEST_NAME,
    DEFAULT_BUY_BUDGET,
    DEFAULT_END_DATE,
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_RANDOM_SEED,
    DEFAULT_STAMP_DUTY_RATE,
    DEFAULT_SELECTION_COOLDOWN_TRADING_DAYS,
    DEFAULT_START_DATE,
    STOP_LOSS_SERIES,
    STOP_LOSS_RULE_NAMES,
    PortfolioBacktestConfig,
    exit_rule_text,
    filter_selection_candidates,
    stop_loss_pct_from_rule_name,
    stop_loss_rule_name,
)
from .db import clear_backtest_rows, connect, ensure_portfolio_tables, load_daily_for_simulation, load_market_trade_dates, save_results
from .simulator import simulate_portfolio


POOL_SOURCES = ("stockselection", "blackbox_predictions", "portfolio_backtest", "csv")


def parse_date(value: str):
    return datetime.strptime(value.replace("-", ""), "%Y%m%d").date()


def load_stockselection_pool(conn, pool_strategy_name: str, start_date, end_date, limit_per_day: int | None) -> pd.DataFrame:
    sql = """
        SELECT
            ss.TradeDate,
            ss.SCode,
            ss.SName,
            ss.ClosePrice AS Close,
            ss.Score,
            ss.Reason,
            ss.StrategyName
        FROM stockselection ss
        WHERE ss.StrategyName = %s
          AND ss.TradeDate >= %s
          AND ss.TradeDate <= %s
        ORDER BY ss.TradeDate, ss.Score DESC, ss.SCode
    """
    with conn.cursor() as cur:
        cur.execute(sql, (pool_strategy_name, start_date, end_date))
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"])
    return limit_pool(frame, limit_per_day)


def load_blackbox_prediction_pool(conn, pool_strategy_name: str, start_date, end_date, limit_per_day: int | None) -> pd.DataFrame:
    sql = """
        SELECT
            p.TradeDate,
            p.SCode,
            si.SName,
            NULL AS Close,
            p.PositiveProbability AS Score,
            CONCAT(
                p.StrategyName,
                ': rank=', p.RankNo,
                '; probability=', p.PositiveProbability,
                '; positive_loss=', IFNULL(p.PositiveLoss, 0),
                '; negative_loss=', IFNULL(p.NegativeLoss, 0)
            ) AS Reason,
            p.StrategyName
        FROM blackbox_predictions p
        LEFT JOIN stockinfo si ON si.SCode = p.SCode
        WHERE p.StrategyName = %s
          AND p.TradeDate >= %s
          AND p.TradeDate <= %s
        ORDER BY p.TradeDate, p.RankNo, p.SCode
    """
    with conn.cursor() as cur:
        cur.execute(sql, (pool_strategy_name, start_date, end_date))
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"])
    return limit_pool(frame, limit_per_day)


def load_portfolio_backtest_pool(
    conn,
    source_backtest_name: str,
    pool_strategy_name: str,
    start_date,
    end_date,
    limit_per_day: int | None,
    source_strategy_name: str | None = None,
    source_trade_rule: str | None = None,
) -> pd.DataFrame:
    if not source_backtest_name:
        raise ValueError("--source-backtest-name is required when --pool-source portfolio_backtest")

    where = [
        "t.BacktestName = %s",
        "t.Side = 'BUY'",
        "t.SelectionDate >= %s",
        "t.SelectionDate <= %s",
    ]
    params: list[object] = [source_backtest_name, start_date, end_date]
    if source_strategy_name:
        where.append("t.StrategyName = %s")
        params.append(source_strategy_name)
    if source_trade_rule:
        where.append("t.TradeRuleName = %s")
        params.append(source_trade_rule)

    sql = f"""
        SELECT
            t.SelectionDate AS TradeDate,
            t.SCode,
            MAX(t.SName) AS SName,
            NULL AS Close,
            NULL AS Score,
            CONCAT(
                'source_backtest=', t.BacktestName,
                '; first_buy_date=', MIN(t.TradeDate),
                '; source_trade_rules=', COUNT(DISTINCT t.TradeRuleName)
            ) AS Reason,
            %s AS StrategyName
        FROM portfolio_backtest_trades t
        WHERE {" AND ".join(where)}
        GROUP BY t.SelectionDate, t.SCode
        ORDER BY t.SelectionDate, MIN(t.TradeDate), t.SCode
    """
    with conn.cursor() as cur:
        cur.execute(sql, [pool_strategy_name, *params])
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"])
    return limit_pool(frame, limit_per_day)


def load_csv_pool(path: str, pool_strategy_name: str, limit_per_day: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename = {
        "trade_date": "TradeDate",
        "date": "TradeDate",
        "scode": "SCode",
        "code": "SCode",
        "sname": "SName",
        "name": "SName",
        "score": "Score",
        "reason": "Reason",
    }
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns})
    if "TradeDate" not in frame.columns or "SCode" not in frame.columns:
        raise ValueError("CSV pool must contain TradeDate/SCode columns, or date/code aliases")
    for column in ("SName", "Score", "Reason"):
        if column not in frame.columns:
            frame[column] = None
    frame["StrategyName"] = pool_strategy_name
    return limit_pool(frame[["TradeDate", "SCode", "SName", "Score", "Reason", "StrategyName"]], limit_per_day)


def limit_pool(frame: pd.DataFrame, limit_per_day: int | None) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.date
    frame["SCode"] = frame["SCode"].astype(str).str.zfill(6)
    frame["Score"] = pd.to_numeric(frame["Score"], errors="coerce")
    if limit_per_day and limit_per_day > 0:
        frame = frame.sort_values(["TradeDate", "Score", "SCode"], ascending=[True, False, True])
        frame = frame.groupby("TradeDate", group_keys=False).head(limit_per_day)
    return frame.reset_index(drop=True)


def load_pool(
    conn,
    source: str,
    pool_strategy_name: str,
    start_date,
    end_date,
    limit_per_day: int | None,
    csv_path: str | None,
    source_backtest_name: str | None = None,
    source_strategy_name: str | None = None,
    source_trade_rule: str | None = None,
) -> pd.DataFrame:
    if source == "stockselection":
        return load_stockselection_pool(conn, pool_strategy_name, start_date, end_date, limit_per_day)
    if source == "blackbox_predictions":
        return load_blackbox_prediction_pool(conn, pool_strategy_name, start_date, end_date, limit_per_day)
    if source == "portfolio_backtest":
        return load_portfolio_backtest_pool(
            conn,
            source_backtest_name or "",
            pool_strategy_name,
            start_date,
            end_date,
            limit_per_day,
            source_strategy_name=source_strategy_name,
            source_trade_rule=source_trade_rule,
        )
    if source == "csv":
        if not csv_path:
            raise ValueError("--csv-path is required when --pool-source csv")
        frame = load_csv_pool(csv_path, pool_strategy_name, limit_per_day)
        frame = frame[(frame["TradeDate"] >= start_date) & (frame["TradeDate"] <= end_date)].copy()
        return frame.reset_index(drop=True)
    raise ValueError(f"unsupported pool source: {source}")


def config_from_args(args, stop_loss_pct: float, trade_rule_name: str) -> PortfolioBacktestConfig:
    pool_label = args.pool_strategy_name.replace(" ", "_")
    return PortfolioBacktestConfig(
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        strategy_name=args.output_strategy_name or pool_label,
        initial_cash=args.initial_cash,
        buy_budget=args.buy_budget,
        fee_rate=args.fee_rate,
        stamp_duty_rate=args.stamp_duty_rate,
        random_seed=args.random_seed,
        backtest_name=args.backtest_name,
        trade_rule_name=trade_rule_name,
        stop_loss_pct=stop_loss_pct,
        selection_rule=(
            f"Use stock pool source={args.pool_source}, pool_strategy={args.pool_strategy_name}; "
            "exclude ST/PT stocks and stocks selected during the previous "
            f"{max(0, args.selection_cooldown_trading_days)} trading days; "
            "buy selected stocks on next trading day at weighted average price."
        ),
        exit_rule=exit_rule_text(stop_loss_pct),
        limit_per_day=args.limit_per_day,
        batch_size=max(1, args.batch_size),
        selection_cooldown_trading_days=max(0, args.selection_cooldown_trading_days),
    )


def run_one(args, stop_loss_pct: float, trade_rule_name: str) -> tuple[int, int, int]:
    config = config_from_args(args, stop_loss_pct, trade_rule_name)
    with connect() as conn:
        ensure_portfolio_tables(conn)
        signals = load_pool(
            conn,
            args.pool_source,
            args.pool_strategy_name,
            config.start_date,
            config.end_date,
            None,
            args.csv_path,
            source_backtest_name=args.source_backtest_name,
            source_strategy_name=args.source_strategy_name,
            source_trade_rule=args.source_trade_rule,
        )
        trade_dates = load_market_trade_dates(conn, config.start_date, config.end_date, config.ktype)
        signals = filter_selection_candidates(
            signals,
            trade_dates,
            cooldown_trading_days=config.selection_cooldown_trading_days,
            limit_per_day=args.limit_per_day,
        )
        print(
            f"pool loaded source={args.pool_source} strategy={args.pool_strategy_name} rows={len(signals)} "
            f"dates={signals['TradeDate'].nunique() if not signals.empty else 0}",
            flush=True,
        )
        daily_df = load_daily_for_simulation(conn, signals, config)
        print(f"daily rows loaded rows={len(daily_df)} symbols={daily_df['SCode'].nunique() if not daily_df.empty else 0}", flush=True)
        daily, holdings, trades = simulate_portfolio(signals, daily_df, config, verbose=not args.quiet)
        if not args.no_save_db:
            if not args.keep_existing:
                clear_backtest_rows(conn, config.backtest_name, config.strategy_name, config.trade_rule_name)
            counts = save_results(conn, daily, holdings, trades)
            print(f"saved daily={counts['daily']} holdings={counts['holdings']} trades={counts['trades']}", flush=True)
        return len(daily), len(holdings), len(trades)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest an existing stock pool with configurable trading rules")
    parser.add_argument("--pool-source", choices=POOL_SOURCES, default="blackbox_predictions")
    parser.add_argument("--pool-strategy-name", required=True, help="Strategy name in the stock pool table, e.g. blackbox_finetune_recall60")
    parser.add_argument("--csv-path", help="CSV path when --pool-source csv")
    parser.add_argument("--source-backtest-name", help="Source BacktestName when --pool-source portfolio_backtest")
    parser.add_argument("--source-strategy-name", help="Optional source StrategyName filter when --pool-source portfolio_backtest")
    parser.add_argument("--source-trade-rule", choices=STOP_LOSS_RULE_NAMES, help="Optional source TradeRuleName filter when --pool-source portfolio_backtest")
    parser.add_argument("--output-strategy-name", help="StrategyName written to portfolio backtest tables; defaults to pool strategy name")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--buy-budget", type=float, default=DEFAULT_BUY_BUDGET)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--stamp-duty-rate", type=float, default=DEFAULT_STAMP_DUTY_RATE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--backtest-name", default=DEFAULT_BACKTEST_NAME)
    parser.add_argument("--trade-rule", choices=STOP_LOSS_RULE_NAMES, default=stop_loss_rule_name(0.03), help="Trading rule used to run this pool backtest.")
    parser.add_argument("--trade-rule-name", default=None, help="Deprecated alias for --trade-rule.")
    parser.add_argument("--stop-loss-pct", type=float, default=0.03)
    parser.add_argument("--trade-rule-series", choices=["single", "stop_loss"], default="single")
    parser.add_argument("--limit-per-day", type=int, default=None)
    parser.add_argument(
        "--selection-cooldown-trading-days",
        type=int,
        default=DEFAULT_SELECTION_COOLDOWN_TRADING_DAYS,
    )
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--no-save-db", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.trade_rule_series == "stop_loss":
        rule_items = [(stop_loss_pct, stop_loss_rule_name(stop_loss_pct)) for stop_loss_pct in STOP_LOSS_SERIES]
    else:
        trade_rule = args.trade_rule_name or args.trade_rule
        stop_loss_pct = stop_loss_pct_from_rule_name(trade_rule) if trade_rule else args.stop_loss_pct
        rule_items = [(stop_loss_pct, trade_rule or stop_loss_rule_name(stop_loss_pct))]
    for stop_loss_pct, trade_rule_name in rule_items:
        daily_count, holding_count, trade_count = run_one(args, stop_loss_pct, trade_rule_name)
        print(
            f"done trade_rule={trade_rule_name} stop_loss={stop_loss_pct * 100:g}% snapshots={daily_count} holdings={holding_count} trades={trade_count}",
            flush=True,
        )


if __name__ == "__main__":
    main()
