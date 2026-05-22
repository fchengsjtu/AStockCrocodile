from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune.prediction_store import first_prediction_date, latest_prediction_date, load_prediction_signals

from .common import (
    BLACKBOX_STRATEGIES,
    DEFAULT_BUY_BUDGET,
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_RANDOM_SEED,
    PortfolioBacktestConfig,
    exit_rule_text,
    stop_loss_rule_name,
)
from .db import clear_backtest_rows, connect, ensure_portfolio_tables, load_daily_for_simulation, save_results
from .simulator import simulate_portfolio


DEFAULT_TRACK_NAME = "blackbox_prediction_tracker_v1"


def parse_date(value: str):
    return datetime.strptime(value.replace("-", ""), "%Y%m%d").date()


def latest_daily_date(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(DATE(KTime)) FROM dkandles WHERE KType = 'D'")
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track black-box prediction portfolios using saved top-N predictions")
    parser.add_argument("--strategy-name", required=True, choices=BLACKBOX_STRATEGIES)
    parser.add_argument("--start-date", help="Defaults to the first saved prediction date for the strategy.")
    parser.add_argument("--end-date", help="Defaults to the latest daily K-line date.")
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--buy-budget", type=float, default=DEFAULT_BUY_BUDGET)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--backtest-name", default=DEFAULT_TRACK_NAME)
    parser.add_argument("--stop-loss-pct", type=float, default=0.03)
    parser.add_argument("--trade-rule-name", default=None)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--no-save-db", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def run_tracker(args) -> tuple[int, int, int]:
    with connect() as conn:
        ensure_portfolio_tables(conn)
        first_date = first_prediction_date(conn, args.strategy_name)
        if first_date is None:
            raise RuntimeError(f"No saved blackbox predictions found for strategy {args.strategy_name}. Run predict_day first.")
        start_date = parse_date(args.start_date) if args.start_date else first_date
        end_date = parse_date(args.end_date) if args.end_date else latest_daily_date(conn)
        if end_date is None:
            raise RuntimeError("No daily K-line rows found in dkandles.")
        last_signal_date = latest_prediction_date(conn, args.strategy_name)
        if last_signal_date is not None:
            end_date = max(end_date, last_signal_date + timedelta(days=15))
        trade_rule_name = args.trade_rule_name or stop_loss_rule_name(args.stop_loss_pct)
        config = PortfolioBacktestConfig(
            start_date=start_date,
            end_date=end_date,
            strategy_name=args.strategy_name,
            initial_cash=args.initial_cash,
            buy_budget=args.buy_budget,
            fee_rate=args.fee_rate,
            random_seed=args.random_seed,
            backtest_name=args.backtest_name,
            trade_rule_name=trade_rule_name,
            stop_loss_pct=args.stop_loss_pct,
            exit_rule=exit_rule_text(args.stop_loss_pct),
            batch_size=max(1, args.batch_size),
        )
        signals = load_prediction_signals(conn, args.strategy_name, start_date, end_date)
        print(f"tracking signals strategy={args.strategy_name} rows={len(signals)} start={start_date} end={end_date}", flush=True)
        daily_df = load_daily_for_simulation(conn, signals, config)
        print(f"tracking daily rows={len(daily_df)} symbols={daily_df['SCode'].nunique() if not daily_df.empty else 0}", flush=True)
        daily, holdings, trades = simulate_portfolio(signals, daily_df, config, verbose=not args.quiet)
        if not args.no_save_db:
            if not args.keep_existing:
                clear_backtest_rows(conn, config.backtest_name, config.strategy_name, config.trade_rule_name)
            counts = save_results(conn, daily, holdings, trades)
            print(f"saved daily={counts['daily']} holdings={counts['holdings']} trades={counts['trades']}", flush=True)
        return len(daily), len(holdings), len(trades)


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    daily_count, holding_count, trade_count = run_tracker(args)
    print(f"done tracker snapshots={daily_count} holdings={holding_count} trades={trade_count}", flush=True)


if __name__ == "__main__":
    main()
