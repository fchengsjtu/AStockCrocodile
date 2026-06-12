from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from .common import PortfolioBacktestConfig, buy_shares_for_budget, round_cent, stop_loss_rule_name, weighted_average_price


@dataclass
class Position:
    scode: str
    sname: str | None
    selection_date: date
    buy_date: date
    buy_index: int
    shares: int
    original_shares: int
    cost_price: float
    strategy_name: str
    score: float | None = None
    reason: str | None = None
    tp10_done: bool = False


def normalize_daily_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    frame = daily_df.copy()
    if frame.empty:
        return frame
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.date
    for column in ("Open", "Close", "High", "Low", "Amount", "Volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["TradeDate", "SCode"]).reset_index(drop=True)


def normalize_signals(signals: pd.DataFrame) -> pd.DataFrame:
    frame = signals.copy()
    if frame.empty:
        return frame
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.date
    if "StrategyName" not in frame.columns:
        frame["StrategyName"] = "unknown"
    if "Score" not in frame.columns:
        frame["Score"] = None
    if "Reason" not in frame.columns:
        frame["Reason"] = None
    if "SName" not in frame.columns:
        frame["SName"] = None
    return frame.sort_values(["TradeDate", "SCode"]).reset_index(drop=True)


def build_symbol_frames(daily_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if daily_df.empty or "SCode" not in daily_df.columns:
        return {}
    frames = {}
    for symbol, group in daily_df.groupby("SCode", sort=False):
        item = group.sort_values("TradeDate").reset_index(drop=True)
        item["TradeIndex"] = range(len(item))
        frames[str(symbol)] = item
    return frames


def next_symbol_row(symbol_frames: dict[str, pd.DataFrame], symbol: str, after_date: date):
    frame = symbol_frames.get(symbol)
    if frame is None or frame.empty:
        return None
    matches = frame[frame["TradeDate"] > after_date]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_buy_schedule(signals: pd.DataFrame, symbol_frames: dict[str, pd.DataFrame]) -> dict[date, list[dict]]:
    schedule: dict[date, list[dict]] = {}
    for row in signals.itertuples(index=False):
        symbol = str(row.SCode)
        buy_row = next_symbol_row(symbol_frames, symbol, row.TradeDate)
        if buy_row is None:
            continue
        buy_date = buy_row.TradeDate
        schedule.setdefault(buy_date, []).append(
            {
                "signal": row,
                "buy_row": buy_row,
                "buy_index": int(buy_row.TradeIndex),
            }
        )
    return schedule


def row_by_date(symbol_frames: dict[str, pd.DataFrame], symbol: str, trade_date: date):
    frame = symbol_frames.get(symbol)
    if frame is None or frame.empty:
        return None
    matches = frame[frame["TradeDate"] == trade_date]
    if matches.empty:
        return None
    return matches.iloc[0]


def mark_price(symbol_frames: dict[str, pd.DataFrame], symbol: str, trade_date: date, fallback: float) -> float:
    row = row_by_date(symbol_frames, symbol, trade_date)
    if row is None or pd.isna(row.Close):
        return fallback
    return float(row.Close)


def trade_record(
    config: PortfolioBacktestConfig,
    trade_date: date,
    position: Position,
    side: str,
    shares: int,
    price: float,
    fee: float,
    stamp_duty: float,
    reason: str,
) -> dict:
    gross = shares * price
    return {
        "BacktestName": config.backtest_name,
        "TradeDate": trade_date,
        "SCode": position.scode,
        "SName": position.sname,
        "StrategyName": position.strategy_name,
        "TradeRuleName": config.trade_rule_name,
        "SelectionDate": position.selection_date,
        "Side": side,
        "Shares": shares,
        "Price": price,
        "GrossAmount": gross,
        "Fee": fee,
        "StampDuty": stamp_duty,
        "NetAmount": gross + fee if side == "BUY" else gross - fee,
        "Reason": reason,
        "SelectionRule": config.selection_rule,
        "ExitRule": config.exit_rule,
    }


def sell_position(
    config: PortfolioBacktestConfig,
    trade_date: date,
    position: Position,
    shares: int,
    price: float,
    reason: str,
) -> tuple[float, float, dict]:
    shares = min(shares, position.shares)
    price = round_cent(price)
    gross = shares * price
    commission = round_cent(gross * config.fee_rate)
    stamp_duty = round_cent(gross * config.stamp_duty_rate)
    fee = commission + stamp_duty
    position.shares -= shares
    cash_delta = gross - fee
    record = trade_record(config, trade_date, position, "SELL", shares, price, fee, stamp_duty, reason)
    return cash_delta, fee, record


def process_position_exit(
    config: PortfolioBacktestConfig,
    trade_date: date,
    position: Position,
    row,
) -> tuple[float, float, list[dict]]:
    if trade_date <= position.buy_date:
        return 0.0, 0.0, []
    elapsed = int(row.TradeIndex) - position.buy_index
    if elapsed < 1 or elapsed > 3:
        return 0.0, 0.0, []

    trades = []
    cash_delta = 0.0
    fee_total = 0.0
    take_10 = round_cent(position.cost_price * 1.10)
    take_20 = round_cent(position.cost_price * 1.20)

    if abs(config.stop_loss_pct - 0.03) < 1e-12:
        intraday_stop_price = round_cent(position.cost_price * 0.95)
        close_stop_price = round_cent(position.cost_price * 0.97)
        if float(row.Open) <= intraday_stop_price and position.shares > 0:
            cash, fee, trade = sell_position(
                config,
                trade_date,
                position,
                position.shares,
                float(row.Open),
                "gap_open_stop_loss_5pct",
            )
            return cash, fee, [trade]
        if float(row.Low) <= intraday_stop_price and position.shares > 0:
            cash, fee, trade = sell_position(
                config,
                trade_date,
                position,
                position.shares,
                intraday_stop_price,
                "intraday_stop_loss_5pct",
            )
            return cash, fee, [trade]
        if float(row.Close) <= close_stop_price and position.shares > 0:
            cash, fee, trade = sell_position(
                config,
                trade_date,
                position,
                position.shares,
                float(row.Close),
                "close_stop_loss_3pct",
            )
            return cash, fee, [trade]
    else:
        stop_price = round_cent(position.cost_price * (1.0 - config.stop_loss_pct))
        if float(row.Low) <= stop_price and position.shares > 0:
            reason = stop_loss_rule_name(config.stop_loss_pct).removesuffix("_take_profit_10_20_hold_3d")
            cash, fee, trade = sell_position(config, trade_date, position, position.shares, stop_price, reason)
            return cash, fee, [trade]

    if not position.tp10_done and float(row.High) >= take_10 and position.shares > 0:
        half = max(1, position.original_shares // 2)
        cash, fee, trade = sell_position(config, trade_date, position, half, take_10, "take_profit_10pct_half")
        cash_delta += cash
        fee_total += fee
        trades.append(trade)
        position.tp10_done = True

    if float(row.High) >= take_20 and position.shares > 0:
        cash, fee, trade = sell_position(config, trade_date, position, position.shares, take_20, "take_profit_20pct_remaining")
        cash_delta += cash
        fee_total += fee
        trades.append(trade)

    if elapsed >= 3 and position.shares > 0:
        cash, fee, trade = sell_position(config, trade_date, position, position.shares, float(row.Close), "time_exit_day3_close")
        cash_delta += cash
        fee_total += fee
        trades.append(trade)

    return cash_delta, fee_total, trades


def build_snapshot(
    config: PortfolioBacktestConfig,
    trade_date: date,
    cash: float,
    positions: list[Position],
    symbol_frames: dict[str, pd.DataFrame],
    daily_buy_amount: float,
    daily_sell_amount: float,
    daily_fee: float,
) -> tuple[dict, list[dict]]:
    holding_rows = []
    holding_value = 0.0
    for position in positions:
        close_price = mark_price(symbol_frames, position.scode, trade_date, position.cost_price)
        market_value = position.shares * close_price
        holding_value += market_value
        holding_rows.append(
            {
                "BacktestName": config.backtest_name,
                "TradeDate": trade_date,
                "SCode": position.scode,
                "SName": position.sname,
                "StrategyName": position.strategy_name,
                "TradeRuleName": config.trade_rule_name,
                "SelectionDate": position.selection_date,
                "BuyDate": position.buy_date,
                "Shares": position.shares,
                "CostPrice": position.cost_price,
                "ClosePrice": close_price,
                "MarketValue": market_value,
                "UnrealizedPnl": (close_price - position.cost_price) * position.shares,
                "SelectionRule": config.selection_rule,
                "ExitRule": config.exit_rule,
            }
        )
    snapshot = {
        "BacktestName": config.backtest_name,
        "TradeDate": trade_date,
        "StrategyName": config.strategy_name,
        "TradeRuleName": config.trade_rule_name,
        "SelectionRule": config.selection_rule,
        "ExitRule": config.exit_rule,
        "TotalMarketValue": cash + holding_value,
        "HoldingMarketValue": holding_value,
        "CashAmount": cash,
        "ActualProfit": cash + holding_value - config.initial_cash,
        "DailyBuyAmount": daily_buy_amount,
        "DailySellAmount": daily_sell_amount,
        "TradingFee": daily_fee,
        "PositionCount": len(positions),
    }
    return snapshot, holding_rows


def simulate_portfolio(
    signals: pd.DataFrame,
    daily_df: pd.DataFrame,
    config: PortfolioBacktestConfig,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_df = normalize_daily_frame(daily_df)
    signals = normalize_signals(signals)
    if daily_df.empty or "SCode" not in daily_df.columns or "TradeDate" not in daily_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    symbol_frames = build_symbol_frames(daily_df)
    buy_schedule = build_buy_schedule(signals, symbol_frames)
    trade_dates = sorted(date_value for date_value in daily_df["TradeDate"].dropna().unique() if date_value >= config.start_date)
    if not trade_dates:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rng = random.Random(config.random_seed)
    cash = float(config.initial_cash)
    positions: list[Position] = []
    snapshots = []
    holdings = []
    trades = []

    for trade_date in trade_dates:
        if trade_date > config.end_date and not positions and not buy_schedule.get(trade_date):
            break

        daily_fee = 0.0
        daily_buy_amount = 0.0
        daily_sell_amount = 0.0

        for position in list(positions):
            row = row_by_date(symbol_frames, position.scode, trade_date)
            if row is None:
                continue
            cash_delta, fee, sell_trades = process_position_exit(config, trade_date, position, row)
            if sell_trades:
                cash += cash_delta
                daily_fee += fee
                daily_sell_amount += sum(item["GrossAmount"] for item in sell_trades)
                trades.extend(sell_trades)
            if position.shares <= 0:
                positions.remove(position)

        candidates = []
        for item in buy_schedule.get(trade_date, []):
            signal = item["signal"]
            buy_row = item["buy_row"]
            price = weighted_average_price(buy_row)
            shares = buy_shares_for_budget(price, config.buy_budget)
            if shares <= 0:
                continue
            gross = shares * price
            fee = round_cent(gross * config.fee_rate)
            candidates.append((item, price, shares, gross, fee, gross + fee))
        if candidates:
            rng.shuffle(candidates)
            for item, price, shares, gross, fee, total_cost in candidates:
                if total_cost > cash:
                    continue
                signal = item["signal"]
                cash -= total_cost
                daily_fee += fee
                daily_buy_amount += gross
                position = Position(
                    scode=str(signal.SCode),
                    sname=getattr(signal, "SName", None),
                    selection_date=signal.TradeDate,
                    buy_date=trade_date,
                    buy_index=int(item["buy_index"]),
                    shares=shares,
                    original_shares=shares,
                    cost_price=price,
                    strategy_name=str(getattr(signal, "StrategyName", config.strategy_name)),
                    score=getattr(signal, "Score", None),
                    reason=getattr(signal, "Reason", None),
                )
                positions.append(position)
                trades.append(
                    trade_record(
                        config,
                        trade_date,
                        position,
                        "BUY",
                        shares,
                        price,
                        fee,
                        0.0,
                        "next_day_average_entry",
                    )
                )

        snapshot, holding_rows = build_snapshot(
            config,
            trade_date,
            cash,
            positions,
            symbol_frames,
            daily_buy_amount,
            daily_sell_amount,
            daily_fee,
        )
        snapshots.append(snapshot)
        holdings.extend(holding_rows)
        if verbose:
            print(
                f"{trade_date} total={snapshot['TotalMarketValue']:.2f} "
                f"holding={snapshot['HoldingMarketValue']:.2f} cash={snapshot['CashAmount']:.2f} "
                f"profit={snapshot['ActualProfit']:.2f} "
                f"fee={daily_fee:.2f} positions={snapshot['PositionCount']}",
                flush=True,
            )

    return pd.DataFrame(snapshots), pd.DataFrame(holdings), pd.DataFrame(trades)
