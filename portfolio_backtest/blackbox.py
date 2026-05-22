from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .common import BLACKBOX_STRATEGIES, PortfolioBacktestConfig


@dataclass(frozen=True)
class BlackboxModules:
    package: str
    common: object
    gpu: object
    inference: object


def package_from_strategy(strategy_name: str) -> str:
    if strategy_name not in BLACKBOX_STRATEGIES:
        raise ValueError(f"unsupported blackbox strategy: {strategy_name}")
    return strategy_name


def load_modules(strategy_name: str) -> BlackboxModules:
    package = package_from_strategy(strategy_name)
    return BlackboxModules(
        package=package,
        common=importlib.import_module(f"{package}.common"),
        gpu=importlib.import_module(f"{package}.gpu"),
        inference=importlib.import_module(f"{package}.inference"),
    )


def load_trade_dates(conn, start_date: date, end_date: date, ktype: str) -> list[date]:
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


def load_symbols_for_date(conn, trade_date: date, ktype: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT SCode
            FROM dkandles
            WHERE KType = %s AND KTime >= %s AND KTime < %s
            ORDER BY SCode
            """,
            (ktype, trade_date, trade_date + timedelta(days=1)),
        )
        rows = cur.fetchall()
    return [row[0] for row in rows]


def load_stock_names(conn, symbols: list[str]) -> dict[str, str | None]:
    if not symbols:
        return {}
    names: dict[str, str | None] = {}
    for start in range(0, len(symbols), 500):
        batch = symbols[start : start + 500]
        placeholders = ",".join(["%s"] * len(batch))
        with conn.cursor() as cur:
            cur.execute(f"SELECT SCode, SName FROM stockinfo WHERE SCode IN ({placeholders})", batch)
            for code, name in cur.fetchall():
                names[str(code)] = name
    return names


def close_from_daily_window(daily: list[dict]) -> float | None:
    if not daily:
        return None
    value = daily[-1].get("Close") if isinstance(daily[-1], dict) else None
    if value is None:
        value = daily[-1].get("close") if isinstance(daily[-1], dict) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_blackbox_signals(conn, config: PortfolioBacktestConfig) -> pd.DataFrame:
    modules = load_modules(config.strategy_name)
    try:
        modules.gpu.prepare_rtx3060(config.blackbox_cuda_device, require_device=not config.blackbox_allow_non_rtx3060)
    except RuntimeError as exc:
        if "PyTorch is required" in str(exc):
            env_name = f".venv-blackbox-finetune-recall{config.strategy_name.removeprefix('blackbox_finetune_recall')}"
            raise RuntimeError(
                f"{config.strategy_name} portfolio backtest requires the blackbox finetune Python environment. "
                f"Run it with {env_name}\\Scripts\\python.exe instead of the main .venv python, or install PyTorch "
                f"and the blackbox requirements into the active environment."
            ) from exc
        raise
    adapter_dir = modules.common.DEFAULT_OUTPUT_DIR / "adapter"
    model, tokenizer = modules.inference.load_model(modules.common.DEFAULT_BASE_MODEL, adapter_dir)

    trade_dates = load_trade_dates(conn, config.start_date, config.end_date, config.ktype)
    rows = []
    for trade_date in trade_dates:
        symbols = load_symbols_for_date(conn, trade_date, config.ktype)
        names = load_stock_names(conn, symbols)
        lookback_start = trade_date - timedelta(days=max(500, config.blackbox_weekly_window * 10, config.blackbox_daily_window * 4))
        batches = list(modules.common.iter_batches(symbols, config.batch_size))
        print(f"{config.strategy_name} predict date={trade_date} symbols={len(symbols)} batches={len(batches)}", flush=True)
        selected_count = 0
        for batch_index, batch in enumerate(batches, start=1):
            daily_map = modules.common.load_kline_map(conn, "dkandles", "D", batch, lookback_start, trade_date)
            weekly_map = modules.common.load_kline_map(conn, "wkandles", "W", batch, lookback_start, trade_date)
            batch_selected = 0
            for scode in batch:
                daily = modules.common.pick_window(daily_map.get(scode, []), trade_date, config.blackbox_daily_window)
                weekly = modules.common.pick_window(weekly_map.get(scode, []), trade_date, config.blackbox_weekly_window)
                if daily is None or weekly is None:
                    continue
                prompt = tokenizer.apply_chat_template(
                    modules.common.build_messages(scode, trade_date, daily, weekly),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                pred = modules.inference.score_prediction(model, tokenizer, prompt, config.blackbox_max_seq_length, config.blackbox_threshold)
                if pred["label"] != "positive":
                    continue
                close_price = close_from_daily_window(daily)
                rows.append(
                    {
                        "TradeDate": trade_date,
                        "SCode": scode,
                        "SName": names.get(scode),
                        "Close": close_price,
                        "Score": pred["positive_probability"],
                        "Reason": (
                            f"{config.strategy_name}: probability={pred['positive_probability']:.6f}; "
                            f"positive_loss={pred['positive_loss']:.6f}; negative_loss={pred['negative_loss']:.6f}"
                        ),
                        "StrategyName": config.strategy_name,
                    }
                )
                batch_selected += 1
                selected_count += 1
            print(f"{config.strategy_name} batch {batch_index}/{len(batches)} selected={batch_selected}", flush=True)
        print(f"{config.strategy_name} date={trade_date} selected={selected_count}", flush=True)

    result = pd.DataFrame(rows, columns=["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"])
    if result.empty:
        return result
    result = result.sort_values(["TradeDate", "Score", "SCode"], ascending=[True, False, True])
    if config.limit_per_day is not None and config.limit_per_day > 0:
        result = result.groupby("TradeDate", group_keys=False).head(config.limit_per_day)
    return result.reset_index(drop=True)
