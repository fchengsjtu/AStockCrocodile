from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP


DEFAULT_BACKTEST_NAME = "portfolio_t1_avg_entry_v1"
DEFAULT_START_DATE = "20260101"
DEFAULT_END_DATE = "20260430"
DEFAULT_INITIAL_CASH = 1_000_000.0
DEFAULT_BUY_BUDGET = 100_000.0
DEFAULT_FEE_RATE = 0.0005
DEFAULT_RANDOM_SEED = 20260519
DEFAULT_SELECTION_RULE = "Use strategy signals on selection date; buy selected stocks on next trading day at same-day weighted average price."
DEFAULT_EXIT_RULE = "T+1 sell rule; within 3 tradable days after buy, stop loss at -3%, take profit half at +10%, take profit remaining half at +20%, otherwise sell remaining shares at day-3 close."
DEFAULT_TRADE_RULE_NAME = "stop_loss_3pct_take_profit_10_20_hold_3d"
DEFAULT_STOP_LOSS_PCT = 0.03
STOP_LOSS_SERIES = (0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06)
BLACKBOX_RECALL_TARGETS = (30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80)
BLACKBOX_STRATEGIES = tuple(f"blackbox_finetune_recall{target}" for target in BLACKBOX_RECALL_TARGETS)


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    start_date: date
    end_date: date
    strategy_name: str
    initial_cash: float = DEFAULT_INITIAL_CASH
    buy_budget: float = DEFAULT_BUY_BUDGET
    fee_rate: float = DEFAULT_FEE_RATE
    random_seed: int = DEFAULT_RANDOM_SEED
    backtest_name: str = DEFAULT_BACKTEST_NAME
    trade_rule_name: str = DEFAULT_TRADE_RULE_NAME
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT
    selection_rule: str = DEFAULT_SELECTION_RULE
    exit_rule: str = DEFAULT_EXIT_RULE
    ktype: str = "D"
    min_turnover_amount: float = 0.0
    limit_per_day: int | None = None
    batch_size: int = 80
    min_recommendations: int = 3
    max_recommendations: int = 5
    blackbox_threshold: float = 0.50
    blackbox_max_seq_length: int = 512
    blackbox_daily_window: int = 55
    blackbox_weekly_window: int = 55
    blackbox_cuda_device: str = "0"
    blackbox_allow_non_rtx3060: bool = False


def is_blackbox_strategy(strategy_name: str) -> bool:
    return strategy_name in BLACKBOX_STRATEGIES


def stop_loss_rule_name(stop_loss_pct: float) -> str:
    basis_points = int(round(stop_loss_pct * 1000))
    if basis_points % 10 == 0:
        label = f"{basis_points // 10}pct"
    else:
        label = f"{basis_points / 10:g}pct".replace(".", "_")
    return f"stop_loss_{label}_take_profit_10_20_hold_3d"


def exit_rule_text(stop_loss_pct: float) -> str:
    percent = stop_loss_pct * 100
    return (
        "T+1 sell rule; within 3 tradable days after buy, "
        f"stop loss at -{percent:g}%, take profit half at +10%, "
        "take profit remaining half at +20%, otherwise sell remaining shares at day-3 close."
    )


def round_cent(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def buy_shares_for_budget(price: float, budget: float) -> int:
    if price <= 0 or budget <= 0:
        return 0
    return max(0, int(round(budget / price / 100.0)) * 100)


def weighted_average_price(row) -> float:
    volume = float(getattr(row, "Volume", 0) or 0)
    amount = float(getattr(row, "Amount", 0) or 0)
    high = float(getattr(row, "High", 0) or 0)
    low = float(getattr(row, "Low", 0) or 0)
    close = float(getattr(row, "Close", 0) or 0)
    if volume > 0 and amount > 0:
        candidates = [amount * 100.0 / volume, amount * 10000.0 / volume]
        lower = low * 0.8 if low > 0 else 0
        upper = high * 1.2 if high > 0 else float("inf")
        valid = [price for price in candidates if lower <= price <= upper]
        if valid:
            anchor = close if close > 0 else (high + low) / 2.0
            return round_cent(min(valid, key=lambda price: abs(price - anchor)))
    return round_cent((high + low + close) / 3.0)
