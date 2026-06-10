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

from stock_selector import STRATEGIES, STRATEGY_MA_BULLISH

from .common import (
    DEFAULT_BACKTEST_NAME,
    DEFAULT_BUY_BUDGET,
    DEFAULT_END_DATE,
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SELECTION_COOLDOWN_TRADING_DAYS,
    DEFAULT_START_DATE,
    BLACKBOX_STRATEGIES,
    STOP_LOSS_SERIES,
    STOP_LOSS_RULE_NAMES,
    PortfolioBacktestConfig,
    exit_rule_text,
    is_blackbox_strategy,
    stop_loss_pct_from_rule_name,
    stop_loss_rule_name,
)
from .blackbox import iter_blackbox_signal_days
from .db import clear_backtest_rows, connect, ensure_portfolio_tables, load_daily_for_simulation, load_strategy_signals, save_results
from .simulator import simulate_portfolio


def parse_date(value: str):
    parsed = datetime.strptime(value.replace("-", ""), "%Y%m%d")
    return parsed.date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run portfolio-level T+1 backtest and save daily holdings into MySQL")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--strategy-name", default=STRATEGY_MA_BULLISH, choices=(*STRATEGIES, *BLACKBOX_STRATEGIES))
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--buy-budget", type=float, default=DEFAULT_BUY_BUDGET)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--backtest-name", default=DEFAULT_BACKTEST_NAME)
    parser.add_argument("--trade-rule", choices=STOP_LOSS_RULE_NAMES, default=None, help="Trading rule name; overrides --stop-loss-pct when set.")
    parser.add_argument("--trade-rule-name", default=None, help="Deprecated alias for --trade-rule.")
    parser.add_argument("--stop-loss-pct", type=float, default=0.03, help="Stop loss percentage as decimal, for example 0.03.")
    parser.add_argument("--trade-rule-series", choices=["single", "stop_loss"], default="single", help="Run one rule or the 3%%-6%% stop-loss rule series.")
    parser.add_argument("--min-turnover-amount", type=float, default=0.0)
    parser.add_argument("--limit-per-day", type=int, default=None)
    parser.add_argument(
        "--selection-cooldown-trading-days",
        type=int,
        default=DEFAULT_SELECTION_COOLDOWN_TRADING_DAYS,
        help="Exclude a selected stock for this many subsequent market trading days.",
    )
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--min-recommendations", type=int, default=3)
    parser.add_argument("--max-recommendations", type=int, default=5)
    parser.add_argument("--blackbox-sample-mode", choices=["short", "long", "xlong", "xxlong"], default="long")
    parser.add_argument("--blackbox-threshold", type=float, default=0.50)
    parser.add_argument("--blackbox-max-seq-length", type=int, default=512)
    parser.add_argument("--blackbox-daily-window", type=int, default=55)
    parser.add_argument("--blackbox-weekly-window", type=int, default=55)
    parser.add_argument("--blackbox-monthly-window", type=int, default=0)
    parser.add_argument("--blackbox-adapter-dir", default=None, help="Override black-box LoRA adapter directory.")
    parser.add_argument("--blackbox-cuda-device", default="0")
    parser.add_argument("--blackbox-allow-non-rtx3060", action="store_true")
    parser.add_argument("--no-save-db", action="store_true")
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete existing rows for this backtest name and strategy before saving.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args) -> PortfolioBacktestConfig:
    trade_rule_arg = args.trade_rule or args.trade_rule_name
    stop_loss_pct = stop_loss_pct_from_rule_name(trade_rule_arg) if trade_rule_arg else args.stop_loss_pct
    trade_rule_name = trade_rule_arg or stop_loss_rule_name(stop_loss_pct)
    cooldown_days = max(0, args.selection_cooldown_trading_days)
    return PortfolioBacktestConfig(
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        strategy_name=args.strategy_name,
        initial_cash=args.initial_cash,
        buy_budget=args.buy_budget,
        fee_rate=args.fee_rate,
        random_seed=args.random_seed,
        backtest_name=args.backtest_name,
        trade_rule_name=trade_rule_name,
        stop_loss_pct=stop_loss_pct,
        selection_rule=(
            "Use strategy signals on selection date; exclude ST/PT stocks and stocks selected "
            f"during the previous {cooldown_days} trading days; buy selected stocks on next "
            "trading day at same-day weighted average price."
        ),
        exit_rule=exit_rule_text(stop_loss_pct),
        min_turnover_amount=args.min_turnover_amount,
        limit_per_day=args.limit_per_day,
        batch_size=args.batch_size,
        min_recommendations=args.min_recommendations,
        max_recommendations=args.max_recommendations,
        selection_cooldown_trading_days=cooldown_days,
        blackbox_sample_mode=args.blackbox_sample_mode,
        blackbox_threshold=args.blackbox_threshold,
        blackbox_max_seq_length=max(64, args.blackbox_max_seq_length),
        blackbox_daily_window=max(2, args.blackbox_daily_window),
        blackbox_weekly_window=max(2, args.blackbox_weekly_window),
        blackbox_monthly_window=max(0, args.blackbox_monthly_window),
        blackbox_adapter_dir=args.blackbox_adapter_dir,
        blackbox_cuda_device=args.blackbox_cuda_device,
        blackbox_allow_non_rtx3060=args.blackbox_allow_non_rtx3060,
    )


def rule_configs_from_args(args) -> list[PortfolioBacktestConfig]:
    if args.trade_rule_series == "single":
        return [config_from_args(args)]
    configs = []
    for stop_loss_pct in STOP_LOSS_SERIES:
        item = argparse.Namespace(**vars(args))
        item.stop_loss_pct = stop_loss_pct
        item.trade_rule = None
        item.trade_rule_name = stop_loss_rule_name(stop_loss_pct)
        configs.append(config_from_args(item))
    return configs


def run(config: PortfolioBacktestConfig, save_db: bool, keep_existing: bool, verbose: bool, signals=None, daily_df=None) -> tuple[int, int, int]:
    with connect() as conn:
        ensure_portfolio_tables(conn)
        if signals is None:
            signals = load_strategy_signals(conn, config)
            print(f"signals loaded strategy={config.strategy_name} rows={len(signals)}", flush=True)
        if daily_df is None:
            daily_df = load_daily_for_simulation(conn, signals, config)
            print(f"daily rows loaded rows={len(daily_df)} symbols={daily_df['SCode'].nunique() if not daily_df.empty else 0}", flush=True)
        print(f"running trade_rule={config.trade_rule_name} stop_loss={config.stop_loss_pct * 100:g}%", flush=True)
        daily, holdings, trades = simulate_portfolio(signals, daily_df, config, verbose=verbose)
        if save_db:
            if not keep_existing:
                clear_backtest_rows(conn, config.backtest_name, config.strategy_name, config.trade_rule_name)
            counts = save_results(conn, daily, holdings, trades)
            print(f"saved daily={counts['daily']} holdings={counts['holdings']} trades={counts['trades']}", flush=True)
        return len(daily), len(holdings), len(trades)


def run_streaming_blackbox(config: PortfolioBacktestConfig, save_db: bool, keep_existing: bool, verbose: bool) -> tuple[int, int, int]:
    if keep_existing and save_db:
        raise ValueError("--keep-existing is not supported for streaming blackbox portfolio backtest")
    with connect() as conn:
        ensure_portfolio_tables(conn)
        if save_db:
            clear_backtest_rows(conn, config.backtest_name, config.strategy_name, config.trade_rule_name)
        signal_frames = []
        latest_counts = (0, 0, 0)
        for trade_date, day_signals in iter_blackbox_signal_days(conn, config):
            if not day_signals.empty:
                signal_frames.append(day_signals)
            if signal_frames:
                signals = pd.concat(signal_frames, ignore_index=True)
            else:
                signals = day_signals
            print(
                f"streaming backtest date={trade_date} day_signals={len(day_signals)} accumulated_signals={len(signals)}",
                flush=True,
            )
            daily_df = load_daily_for_simulation(conn, signals, config)
            daily, holdings, trades = simulate_portfolio(signals, daily_df, config, verbose=verbose)
            if save_db:
                clear_backtest_rows(conn, config.backtest_name, config.strategy_name, config.trade_rule_name)
                counts = save_results(conn, daily, holdings, trades)
                print(
                    f"streaming saved through_signal_date={trade_date} daily={counts['daily']} holdings={counts['holdings']} trades={counts['trades']}",
                    flush=True,
                )
            latest_counts = (len(daily), len(holdings), len(trades))
        return latest_counts


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configs = rule_configs_from_args(args)
    if len(configs) == 1 and is_blackbox_strategy(configs[0].strategy_name):
        daily_count, holding_count, trade_count = run_streaming_blackbox(
            configs[0],
            save_db=not args.no_save_db,
            keep_existing=args.keep_existing,
            verbose=not args.quiet,
        )
        print(f"done rule={configs[0].trade_rule_name} snapshots={daily_count} holdings={holding_count} trades={trade_count}", flush=True)
        return
    shared_signals = None
    shared_daily = None
    if configs:
        with connect() as conn:
            ensure_portfolio_tables(conn)
            shared_signals = load_strategy_signals(conn, configs[0])
            print(f"signals loaded strategy={configs[0].strategy_name} rows={len(shared_signals)}", flush=True)
            shared_daily = load_daily_for_simulation(conn, shared_signals, configs[0])
            print(f"daily rows loaded rows={len(shared_daily)} symbols={shared_daily['SCode'].nunique() if not shared_daily.empty else 0}", flush=True)
    for config in configs:
        daily_count, holding_count, trade_count = run(
            config,
            save_db=not args.no_save_db,
            keep_existing=args.keep_existing,
            verbose=not args.quiet,
            signals=shared_signals,
            daily_df=shared_daily,
        )
        print(f"done rule={config.trade_rule_name} snapshots={daily_count} holdings={holding_count} trades={trade_count}", flush=True)


if __name__ == "__main__":
    main()
