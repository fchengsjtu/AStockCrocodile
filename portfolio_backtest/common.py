from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd


DEFAULT_BACKTEST_NAME = "portfolio_t1_avg_entry_v1"
DEFAULT_START_DATE = "20260101"
DEFAULT_END_DATE = "20260430"
DEFAULT_INITIAL_CASH = 1_000_000.0
DEFAULT_BUY_BUDGET = 100_000.0
DEFAULT_FEE_RATE = 0.0002
DEFAULT_RANDOM_SEED = 20260519
DEFAULT_SELECTION_COOLDOWN_TRADING_DAYS = 13
DEFAULT_SELECTION_RULE = (
    "Use strategy signals on selection date; exclude ST/PT stocks and stocks selected "
    "during the previous 13 trading days; buy selected stocks on next trading day at "
    "same-day weighted average price."
)
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
    selection_cooldown_trading_days: int = DEFAULT_SELECTION_COOLDOWN_TRADING_DAYS
    blackbox_sample_mode: str = "long"
    blackbox_threshold: float = 0.50
    blackbox_max_seq_length: int = 512
    blackbox_daily_window: int = 55
    blackbox_weekly_window: int = 55
    blackbox_monthly_window: int = 0
    blackbox_adapter_dir: str | None = None
    blackbox_cuda_device: str = "0"
    blackbox_allow_non_rtx3060: bool = False


def is_blackbox_strategy(strategy_name: str) -> bool:
    return strategy_name in BLACKBOX_STRATEGIES


def is_special_treatment_stock_name(value) -> bool:
    if value is None or pd.isna(value):
        return False
    normalized = re.sub(r"\s+", "", str(value)).upper()
    return "ST" in normalized or "PT" in normalized


def filter_selection_candidates(
    frame: pd.DataFrame,
    trade_dates,
    cooldown_trading_days: int = DEFAULT_SELECTION_COOLDOWN_TRADING_DAYS,
    limit_per_day: int | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["TradeDate"] = pd.to_datetime(result["TradeDate"]).dt.date
    result["SCode"] = result["SCode"].astype(str).str.zfill(6)
    if "SName" in result.columns:
        result = result[~result["SName"].map(is_special_treatment_stock_name)].copy()
    if result.empty:
        return result.reset_index(drop=True)
    if "Score" not in result.columns:
        result["Score"] = None
    result["Score"] = pd.to_numeric(result["Score"], errors="coerce")
    result = result.sort_values(
        ["TradeDate", "Score", "SCode"],
        ascending=[True, False, True],
        na_position="last",
    )

    calendar = sorted({pd.Timestamp(item).date() for item in trade_dates})
    date_index = {trade_date: index for index, trade_date in enumerate(calendar)}
    last_selected_index: dict[str, int] = {}
    selected_indices: list[int] = []
    cooldown = max(0, int(cooldown_trading_days))
    daily_limit = limit_per_day if limit_per_day is not None and limit_per_day > 0 else None

    for trade_date, day_rows in result.groupby("TradeDate", sort=True):
        current_index = date_index.get(trade_date)
        if current_index is None:
            continue
        selected_today = 0
        for row_index, row in day_rows.iterrows():
            scode = str(row["SCode"])
            previous_index = last_selected_index.get(scode)
            if previous_index is not None and current_index - previous_index <= cooldown:
                continue
            selected_indices.append(row_index)
            last_selected_index[scode] = current_index
            selected_today += 1
            if daily_limit is not None and selected_today >= daily_limit:
                break
    return result.loc[selected_indices].reset_index(drop=True)


def stop_loss_rule_name(stop_loss_pct: float) -> str:
    label = f"{stop_loss_pct * 100:g}pct"
    return f"stop_loss_{label}_take_profit_10_20_hold_3d"


STOP_LOSS_RULE_NAMES = tuple(stop_loss_rule_name(item) for item in STOP_LOSS_SERIES)


def stop_loss_pct_from_rule_name(rule_name: str) -> float:
    match = re.fullmatch(r"stop_loss_(\d+(?:\.\d+)?)pct_take_profit_10_20_hold_3d", rule_name.strip())
    if not match:
        raise ValueError(f"unsupported trade rule: {rule_name}")
    pct = float(match.group(1)) / 100.0
    if pct not in STOP_LOSS_SERIES:
        allowed = ", ".join(STOP_LOSS_RULE_NAMES)
        raise ValueError(f"unsupported stop-loss percentage in {rule_name}; allowed rules: {allowed}")
    return pct


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
