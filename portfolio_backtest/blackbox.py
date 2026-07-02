from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .common import (
    BLACKBOX_STRATEGIES,
    PortfolioBacktestConfig,
    filter_selection_candidates,
    is_special_treatment_stock_name,
)


@dataclass(frozen=True)
class BlackboxModules:
    package: str
    common: object
    gpu: object
    inference: object


def package_from_strategy(strategy_name: str) -> str:
    if strategy_name not in BLACKBOX_STRATEGIES:
        raise ValueError(f"unsupported blackbox strategy: {strategy_name}")
    return "blackbox_finetune_recall60"


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


def candidate_from_prediction(
    config: PortfolioBacktestConfig,
    trade_date: date,
    scode: str,
    sname: str | None,
    close_price: float | None,
    prediction: dict,
) -> dict:
    probability = float(prediction["positive_probability"])
    return {
        "TradeDate": trade_date,
        "SCode": scode,
        "SName": sname,
        "Close": close_price,
        "Score": probability,
        "Reason": (
            f"{config.strategy_name}: probability={probability:.6f}; "
            f"positive_loss={prediction['positive_loss']:.6f}; "
            f"negative_loss={prediction['negative_loss']:.6f}; "
            f"threshold_passed={probability >= config.blackbox_threshold}"
        ),
        "StrategyName": config.strategy_name,
    }


def format_top_predictions(frame: pd.DataFrame, limit: int = 5) -> str:
    if frame.empty:
        return "<none>"
    top_rows = frame.sort_values(
        ["Score", "SCode"],
        ascending=[False, True],
        na_position="last",
    ).head(max(0, limit))
    items = []
    for row in top_rows.itertuples(index=False):
        name = row.SName if pd.notna(row.SName) and row.SName else "<unknown>"
        score = float(row.Score) if pd.notna(row.Score) else float("nan")
        score_text = f"{score:.6f}" if pd.notna(score) else "nan"
        items.append(f"{row.SCode}:{name}:score={score_text}")
    return ",".join(items) or "<none>"


def windows_are_scoreable(
    daily: list[dict] | None,
    weekly: list[dict] | None,
    monthly: list[dict] | None,
    daily_window: int,
    weekly_window: int,
    monthly_window: int,
) -> bool:
    return (
        daily is not None
        and len(daily) >= daily_window
        and weekly is not None
        and len(weekly) >= weekly_window
        and (
            monthly_window <= 0
            or (monthly is not None and len(monthly) >= monthly_window)
        )
    )


def load_blackbox_model(config: PortfolioBacktestConfig):
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
    if config.blackbox_adapter_dir:
        adapter_dir = Path(config.blackbox_adapter_dir)
        if not (adapter_dir / "adapter_config.json").exists() and (adapter_dir / "adapter" / "adapter_config.json").exists():
            adapter_dir = adapter_dir / "adapter"
    else:
        output_dir = modules.common.default_output_dir() if hasattr(modules.common, "default_output_dir") else modules.common.DEFAULT_OUTPUT_DIR
        adapter_dir = output_dir / "adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise RuntimeError(f"blackbox adapter_config.json not found in {adapter_dir}")
    model, tokenizer = modules.inference.load_model(modules.common.DEFAULT_BASE_MODEL, adapter_dir)
    return modules, model, tokenizer


def iter_blackbox_signal_days(conn, config: PortfolioBacktestConfig):
    modules, model, tokenizer = load_blackbox_model(config)
    trade_dates = load_trade_dates(conn, config.start_date, config.end_date, config.ktype)
    selected_history = []
    for trade_date in trade_dates:
        rows = []
        symbols = load_symbols_for_date(conn, trade_date, config.ktype)
        names = load_stock_names(conn, symbols)
        lookback_start = trade_date - timedelta(days=max(750, config.blackbox_monthly_window * 45, config.blackbox_weekly_window * 14, config.blackbox_daily_window * 5))
        batches = list(modules.common.iter_batches(symbols, config.batch_size))
        print(f"{config.strategy_name} predict date={trade_date} symbols={len(symbols)} batches={len(batches)}", flush=True)
        scored_count = 0
        threshold_count = 0
        skipped_name_count = 0
        missing_daily_count = 0
        missing_weekly_count = 0
        missing_monthly_count = 0
        skipped_rule_count = 0
        for batch_index, batch in enumerate(batches, start=1):
            batch_started = perf_counter()
            daily_map = modules.common.load_kline_map(conn, "dkandles", "D", batch, lookback_start, trade_date)
            weekly_map = modules.common.load_kline_map(conn, "wkandles", "W", batch, lookback_start, trade_date)
            monthly_map = (
                modules.common.load_kline_map(conn, "mkandles", "M", batch, lookback_start, trade_date)
                if config.blackbox_monthly_window > 0
                else {}
            )
            batch_scored = 0
            batch_threshold_count = 0
            batch_skipped_name = 0
            batch_missing_daily = 0
            batch_missing_weekly = 0
            batch_missing_monthly = 0
            batch_skipped_rule = 0
            for scode in batch:
                if is_special_treatment_stock_name(names.get(scode)):
                    batch_skipped_name += 1
                    continue
                daily = modules.common.pick_window(daily_map.get(scode, []), trade_date, config.blackbox_daily_window)
                if hasattr(modules.common, "pick_weekly_window"):
                    weekly = modules.common.pick_weekly_window(weekly_map.get(scode, []), daily_map.get(scode, []), trade_date, config.blackbox_weekly_window)
                else:
                    weekly = modules.common.pick_window(weekly_map.get(scode, []), trade_date, config.blackbox_weekly_window)
                if hasattr(modules.common, "pick_monthly_window"):
                    monthly = modules.common.pick_monthly_window(monthly_map.get(scode, []), trade_date, config.blackbox_monthly_window)
                else:
                    monthly = modules.common.pick_window(monthly_map.get(scode, []), trade_date, config.blackbox_monthly_window) if config.blackbox_monthly_window > 0 else []
                if daily is None or len(daily) < config.blackbox_daily_window:
                    batch_missing_daily += 1
                    missing_daily_count += 1
                    continue
                if weekly is None or len(weekly) < config.blackbox_weekly_window:
                    batch_missing_weekly += 1
                    missing_weekly_count += 1
                    continue
                if config.blackbox_monthly_window > 0 and (
                    monthly is None or len(monthly) < config.blackbox_monthly_window
                ):
                    batch_missing_monthly += 1
                    missing_monthly_count += 1
                    continue
                if not windows_are_scoreable(
                    daily,
                    weekly,
                    monthly,
                    config.blackbox_daily_window,
                    config.blackbox_weekly_window,
                    config.blackbox_monthly_window,
                ):
                    continue
                if hasattr(modules.common, "_sample_windows_are_valid") and not modules.common._sample_windows_are_valid(config.blackbox_sample_mode, weekly, monthly, daily):
                    batch_skipped_rule += 1
                    continue
                prompt = tokenizer.apply_chat_template(
                    modules.common.build_messages(scode, trade_date, daily, weekly, monthly, sample_mode=config.blackbox_sample_mode),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                pred = modules.inference.score_prediction(model, tokenizer, prompt, config.blackbox_max_seq_length, config.blackbox_threshold)
                batch_scored += 1
                scored_count += 1
                if float(pred["positive_probability"]) < config.blackbox_threshold:
                    continue
                close_price = close_from_daily_window(daily)
                rows.append(
                    candidate_from_prediction(
                        config,
                        trade_date,
                        scode,
                        names.get(scode),
                        close_price,
                        pred,
                    )
                )
                batch_threshold_count += 1
                threshold_count += 1
            skipped_name_count += batch_skipped_name
            skipped_rule_count += batch_skipped_rule
            elapsed = perf_counter() - batch_started
            print(
                f"{config.strategy_name} batch {batch_index}/{len(batches)} "
                f"scored={batch_scored} above_threshold={batch_threshold_count} "
                f"skipped_name={batch_skipped_name} missing_daily={batch_missing_daily} "
                f"missing_weekly={batch_missing_weekly} missing_monthly={batch_missing_monthly} "
                f"skipped_rule={batch_skipped_rule} elapsed={elapsed:.2f}s",
                flush=True,
            )
        print(
            f"{config.strategy_name} date={trade_date} scored={scored_count} "
            f"above_threshold={threshold_count} missing_daily={missing_daily_count} "
            f"missing_weekly={missing_weekly_count} missing_monthly={missing_monthly_count} "
            f"skipped_name={skipped_name_count} skipped_rule={skipped_rule_count}",
            flush=True,
        )
        result = pd.DataFrame(rows, columns=["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"])
        if not result.empty:
            candidate_history = pd.DataFrame(
                [*selected_history, *result.to_dict("records")],
                columns=result.columns,
            )
            filtered_history = filter_selection_candidates(
                candidate_history,
                trade_dates,
                cooldown_trading_days=config.selection_cooldown_trading_days,
                limit_per_day=config.limit_per_day,
            )
            result = filtered_history[filtered_history["TradeDate"] == trade_date].copy()
            selected_history = filtered_history.to_dict("records")
        print(
            f"{config.strategy_name} date={trade_date} top5={format_top_predictions(result)}",
            flush=True,
        )
        yield trade_date, result.reset_index(drop=True)


def build_blackbox_signals(conn, config: PortfolioBacktestConfig) -> pd.DataFrame:
    rows = []
    for _, frame in iter_blackbox_signal_days(conn, config):
        if not frame.empty:
            rows.extend(frame.to_dict("records"))
    result = pd.DataFrame(rows, columns=["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"])
    return result.reset_index(drop=True) if not result.empty else result
