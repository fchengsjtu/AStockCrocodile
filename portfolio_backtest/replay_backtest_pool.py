from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .common import (
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_RANDOM_SEED,
    DEFAULT_STAMP_DUTY_RATE,
    PortfolioBacktestConfig,
    exit_rule_text,
    filter_selection_candidates,
    stop_loss_pct_from_rule_name,
    stop_loss_rule_name,
)
from .db import connect, ensure_portfolio_tables, load_daily_for_simulation, load_market_trade_dates
from .pool_run import load_portfolio_backtest_pool
from .simulator import simulate_portfolio


def parse_date(value: str):
    return datetime.strptime(value.replace("-", ""), "%Y%m%d").date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay selected stocks from an existing portfolio backtest without writing database rows."
    )
    parser.add_argument("--source-backtest-name", required=True, help="Existing BacktestName used as the stock pool source.")
    parser.add_argument("--source-strategy-name", help="Optional source StrategyName filter.")
    parser.add_argument("--source-trade-rule", help="Optional source TradeRuleName filter.")
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20260630")
    parser.add_argument("--trade-rule", default=stop_loss_rule_name(0.06))
    parser.add_argument("--strategy-name", default="replay_portfolio_backtest_pool")
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--buy-budget", type=float, default=100_000.0)
    parser.add_argument("--limit-per-day", type=int, default=3)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--stamp-duty-rate", type=float, default=DEFAULT_STAMP_DUTY_RATE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--selection-cooldown-trading-days", type=int, default=0)
    parser.add_argument("--quiet-daily", action="store_true")
    return parser


def build_config(args) -> PortfolioBacktestConfig:
    stop_loss_pct = stop_loss_pct_from_rule_name(args.trade_rule)
    return PortfolioBacktestConfig(
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        strategy_name=args.strategy_name,
        initial_cash=args.initial_cash,
        buy_budget=args.buy_budget,
        fee_rate=args.fee_rate,
        stamp_duty_rate=args.stamp_duty_rate,
        random_seed=args.random_seed,
        backtest_name=f"replay_{args.source_backtest_name}",
        trade_rule_name=args.trade_rule,
        stop_loss_pct=stop_loss_pct,
        selection_rule=(
            f"Replay BUY selections from source_backtest={args.source_backtest_name}; "
            f"limit_per_day={args.limit_per_day}; buy_budget={args.buy_budget:g}; no database writes."
        ),
        exit_rule=exit_rule_text(stop_loss_pct),
        limit_per_day=args.limit_per_day,
        batch_size=max(1, args.batch_size),
        selection_cooldown_trading_days=max(0, args.selection_cooldown_trading_days),
    )


def enrich_trade_details(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    frame = trades.copy()
    frame["RealizedPnl"] = 0.0
    frame["BuyCost"] = 0.0
    frame["BuyFeeAllocated"] = 0.0
    frame["TotalFeeAllocated"] = pd.to_numeric(frame["Fee"], errors="coerce").fillna(0.0)

    open_lots: dict[tuple[str, object], deque[dict]] = defaultdict(deque)
    for index, row in frame.iterrows():
        key = (str(row["SCode"]), row["SelectionDate"])
        shares = int(row["Shares"])
        if row["Side"] == "BUY":
            open_lots[key].append(
                {
                    "shares": shares,
                    "price": float(row["Price"]),
                    "fee_per_share": float(row["Fee"]) / shares if shares > 0 else 0.0,
                }
            )
            continue

        remaining = shares
        buy_cost = 0.0
        buy_fee = 0.0
        while remaining > 0 and open_lots[key]:
            lot = open_lots[key][0]
            used = min(remaining, int(lot["shares"]))
            buy_cost += used * float(lot["price"])
            buy_fee += used * float(lot["fee_per_share"])
            lot["shares"] -= used
            remaining -= used
            if lot["shares"] <= 0:
                open_lots[key].popleft()
        sell_gross = float(row["GrossAmount"])
        sell_fee = float(row["Fee"])
        realized = sell_gross - sell_fee - buy_cost - buy_fee
        frame.at[index, "BuyCost"] = buy_cost
        frame.at[index, "BuyFeeAllocated"] = buy_fee
        frame.at[index, "TotalFeeAllocated"] = sell_fee + buy_fee
        frame.at[index, "RealizedPnl"] = realized
    return frame


def print_trade_details(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("No trades generated.", flush=True)
        return
    detailed = enrich_trade_details(trades)
    print("==== trade details ====", flush=True)
    header = [
        "TradeDate",
        "Side",
        "SCode",
        "SName",
        "SelectionDate",
        "Shares",
        "Price",
        "GrossAmount",
        "BuyCost",
        "Fee",
        "StampDuty",
        "TotalFeeAllocated",
        "RealizedPnl",
        "Reason",
    ]
    print("\t".join(header), flush=True)
    for row in detailed.sort_values(["TradeDate", "Side", "SCode"]).itertuples(index=False):
        values = {
            "TradeDate": row.TradeDate,
            "Side": row.Side,
            "SCode": row.SCode,
            "SName": row.SName or "",
            "SelectionDate": row.SelectionDate,
            "Shares": int(row.Shares),
            "Price": f"{float(row.Price):.4f}",
            "GrossAmount": f"{float(row.GrossAmount):.2f}",
            "BuyCost": f"{float(row.BuyCost):.2f}",
            "Fee": f"{float(row.Fee):.2f}",
            "StampDuty": f"{float(row.StampDuty):.2f}",
            "TotalFeeAllocated": f"{float(row.TotalFeeAllocated):.2f}",
            "RealizedPnl": f"{float(row.RealizedPnl):.2f}",
            "Reason": row.Reason,
        }
        print("\t".join(str(values[column]) for column in header), flush=True)


def print_summary(daily: pd.DataFrame, trades: pd.DataFrame, signals: pd.DataFrame) -> None:
    buy_count = int((trades["Side"] == "BUY").sum()) if not trades.empty else 0
    sell_count = int((trades["Side"] == "SELL").sum()) if not trades.empty else 0
    final = daily.iloc[-1].to_dict() if not daily.empty else {}
    print("==== summary ====", flush=True)
    print(f"signals={len(signals)} signal_dates={signals['TradeDate'].nunique() if not signals.empty else 0}", flush=True)
    print(f"buy_trades={buy_count} sell_trades={sell_count}", flush=True)
    if final:
        print(
            "final "
            f"date={final['TradeDate']} "
            f"total={float(final['TotalMarketValue']):.2f} "
            f"holding={float(final['HoldingMarketValue']):.2f} "
            f"cash={float(final['CashAmount']):.2f} "
            f"actual_profit={float(final['ActualProfit']):.2f} "
            f"positions={int(final['PositionCount'])}",
            flush=True,
        )
    if not trades.empty:
        reason_counts = trades.groupby(["Side", "Reason"]).size().reset_index(name="Count")
        print("==== reason counts ====", flush=True)
        for row in reason_counts.itertuples(index=False):
            print(f"{row.Side}\t{row.Reason}\t{row.Count}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = build_config(args)
    with connect() as conn:
        ensure_portfolio_tables(conn)
        signals = load_portfolio_backtest_pool(
            conn,
            args.source_backtest_name,
            args.strategy_name,
            config.start_date,
            config.end_date,
            None,
            source_strategy_name=args.source_strategy_name,
            source_trade_rule=args.source_trade_rule,
        )
        trade_dates = load_market_trade_dates(conn, config.start_date, config.end_date, config.ktype)
        signals = filter_selection_candidates(
            signals,
            trade_dates,
            cooldown_trading_days=config.selection_cooldown_trading_days,
            limit_per_day=config.limit_per_day,
        )
        print(
            f"pool loaded source_backtest={args.source_backtest_name} rows={len(signals)} "
            f"dates={signals['TradeDate'].nunique() if not signals.empty else 0}",
            flush=True,
        )
        daily_df = load_daily_for_simulation(conn, signals, config)
        print(
            f"daily rows loaded rows={len(daily_df)} symbols={daily_df['SCode'].nunique() if not daily_df.empty else 0}",
            flush=True,
        )
    daily, _, trades = simulate_portfolio(signals, daily_df, config, verbose=not args.quiet_daily)
    print_trade_details(trades)
    print_summary(daily, trades, signals)


if __name__ == "__main__":
    main()
