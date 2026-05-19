from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_selector import STRATEGIES, STRATEGY_MA_BULLISH

from .common import (
    DEFAULT_BACKTEST_NAME,
    DEFAULT_BUY_BUDGET,
    DEFAULT_END_DATE,
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_RANDOM_SEED,
    DEFAULT_START_DATE,
    PortfolioBacktestConfig,
)
from .db import clear_backtest_rows, connect, ensure_portfolio_tables, load_daily_for_simulation, load_strategy_signals, save_results
from .simulator import simulate_portfolio


def parse_date(value: str):
    parsed = datetime.strptime(value.replace("-", ""), "%Y%m%d")
    return parsed.date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run portfolio-level T+1 backtest and save daily holdings into MySQL")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--strategy-name", default=STRATEGY_MA_BULLISH, choices=STRATEGIES)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--buy-budget", type=float, default=DEFAULT_BUY_BUDGET)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--backtest-name", default=DEFAULT_BACKTEST_NAME)
    parser.add_argument("--min-turnover-amount", type=float, default=0.0)
    parser.add_argument("--limit-per-day", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--min-recommendations", type=int, default=3)
    parser.add_argument("--max-recommendations", type=int, default=5)
    parser.add_argument("--no-save-db", action="store_true")
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete existing rows for this backtest name and strategy before saving.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args) -> PortfolioBacktestConfig:
    return PortfolioBacktestConfig(
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        strategy_name=args.strategy_name,
        initial_cash=args.initial_cash,
        buy_budget=args.buy_budget,
        fee_rate=args.fee_rate,
        random_seed=args.random_seed,
        backtest_name=args.backtest_name,
        min_turnover_amount=args.min_turnover_amount,
        limit_per_day=args.limit_per_day,
        batch_size=args.batch_size,
        min_recommendations=args.min_recommendations,
        max_recommendations=args.max_recommendations,
    )


def run(config: PortfolioBacktestConfig, save_db: bool, keep_existing: bool, verbose: bool) -> tuple[int, int, int]:
    with connect() as conn:
        ensure_portfolio_tables(conn)
        signals = load_strategy_signals(conn, config)
        print(f"signals loaded strategy={config.strategy_name} rows={len(signals)}", flush=True)
        daily_df = load_daily_for_simulation(conn, signals, config)
        print(f"daily rows loaded rows={len(daily_df)} symbols={daily_df['SCode'].nunique() if not daily_df.empty else 0}", flush=True)
        daily, holdings, trades = simulate_portfolio(signals, daily_df, config, verbose=verbose)
        if save_db:
            if not keep_existing:
                clear_backtest_rows(conn, config.backtest_name, config.strategy_name)
            counts = save_results(conn, daily, holdings, trades)
            print(f"saved daily={counts['daily']} holdings={counts['holdings']} trades={counts['trades']}", flush=True)
        return len(daily), len(holdings), len(trades)


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    daily_count, holding_count, trade_count = run(config, save_db=not args.no_save_db, keep_existing=args.keep_existing, verbose=not args.quiet)
    print(f"done snapshots={daily_count} holdings={holding_count} trades={trade_count}", flush=True)


if __name__ == "__main__":
    main()

