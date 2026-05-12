from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from a_share_crawler import DEFAULT_KTYPE
from llm_blackbox_pattern_trainer import _compact_window_to_matrix
from stock_selector import parse_date
from surge_pattern_miner import (
    extract_features_for_date,
    iter_batches,
    load_kline_for_symbols,
    make_frame_map,
)

DEFAULT_BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_FINETUNE_DIR = Path("llm_finetune") / "runs" / "deepseek-r1-distill-qwen-7b-lora"
DEFAULT_DATA_DIR = Path("llm_finetune") / "data"
DEFAULT_WINDOW = 55
SYSTEM_PROMPT = (
    "You are a stock pattern mining model. Given compact 55 daily and 55 weekly OHLCV windows, "
    "return strict JSON only. Use exact feature tokens from allowed_feature_tokens. "
    "For a tradable setup return label=positive and 3 to 8 feature tokens. Otherwise return label=negative."
)


@dataclass(frozen=True)
class KlineWindowSample:
    scode: str
    trade_date: date
    label: str
    gain_rate: float | None
    features: tuple[str, ...]
    daily_55: dict
    weekly_55: dict


def normalize_rate(value: float) -> float:
    return value / 100 if value > 1 else value


def json_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_instruction(sample: KlineWindowSample, allowed_features: Iterable[str]) -> str:
    payload = {
        "task": "Classify whether this stock/date matches a short-term surge setup and return reusable feature-token rules.",
        "anchor_date": str(sample.trade_date),
        "scode": sample.scode,
        "daily_55": sample.daily_55,
        "weekly_55": sample.weekly_55,
        "allowed_feature_tokens": sorted(set(allowed_features)),
        "output_schema": {
            "label": "positive|negative",
            "patterns": [{"features": ["TOKEN_A", "TOKEN_B", "TOKEN_C"]}],
        },
    }
    return SYSTEM_PROMPT + "\n\n" + json_dumps(payload)


def choose_target_features(features: Iterable[str], min_size: int = 3, max_size: int = 8) -> tuple[str, ...]:
    preferred_prefixes = (
        "D_RET_",
        "W_RET_",
        "D_VOL_",
        "W_VOL_",
        "D_RANGE_",
        "D_CLOSE_",
        "W_CLOSE_",
        "D_MA",
        "W_MA",
    )
    ordered = sorted(set(features), key=lambda item: (next((idx for idx, prefix in enumerate(preferred_prefixes) if item.startswith(prefix)), 99), item))
    if len(ordered) < min_size:
        return tuple(ordered)
    return tuple(ordered[:max_size])


def build_response(sample: KlineWindowSample, min_pattern_size: int = 3, max_pattern_size: int = 8) -> str:
    if sample.label != "positive":
        return json_dumps({"label": "negative", "patterns": []})
    features = choose_target_features(sample.features, min_pattern_size, max_pattern_size)
    if len(features) < min_pattern_size:
        return json_dumps({"label": "negative", "patterns": []})
    return json_dumps(
        {
            "label": "positive",
            "patterns": [
                {
                    "features": list(features),
                    "rationale": "Historical positive sample with matching compact daily and weekly setup.",
                }
            ],
        }
    )


def to_messages_jsonl(sample: KlineWindowSample, allowed_features: Iterable[str]) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_instruction(sample, allowed_features)},
            {"role": "assistant", "content": build_response(sample)},
        ],
        "metadata": {
            "scode": sample.scode,
            "trade_date": str(sample.trade_date),
            "label": sample.label,
            "gain_rate": sample.gain_rate,
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json_dumps(row) + "\n")
            count += 1
    return count


def collect_window_samples(
    conn,
    events: pd.DataFrame,
    label: str,
    daily_window: int,
    weekly_window: int,
    batch_size: int,
) -> tuple[list[KlineWindowSample], set[str]]:
    if events.empty:
        return [], set()
    start_date = min(events["TradeDate"])
    end_date = max(events["TradeDate"])
    lookback_start = start_date - timedelta(days=max(500, weekly_window * 10, daily_window * 3))
    symbols = sorted(events["SCode"].dropna().unique().tolist())
    samples: list[KlineWindowSample] = []
    all_features: set[str] = set()
    for batch_index, batch in enumerate(iter_batches(symbols, batch_size), start=1):
        daily_df = load_kline_for_symbols(conn, "dkandles", DEFAULT_KTYPE, batch, lookback_start, end_date)
        weekly_df = load_kline_for_symbols(conn, "wkandles", "W", batch, lookback_start, end_date)
        daily_frames = make_frame_map(daily_df)
        weekly_frames = make_frame_map(weekly_df)
        batch_events = events[events["SCode"].isin(batch)]
        batch_samples = 0
        for event in batch_events.itertuples(index=False):
            daily_frame = daily_frames.get(event.SCode)
            weekly_frame = weekly_frames.get(event.SCode)
            if daily_frame is None or weekly_frame is None:
                continue
            daily_matches = daily_frame.index[daily_frame["TradeDate"] == event.TradeDate].tolist()
            if not daily_matches:
                continue
            daily_pos = int(daily_matches[0])
            weekly_positions = weekly_frame.index[weekly_frame["TradeDate"] <= event.TradeDate].tolist()
            if not weekly_positions:
                continue
            weekly_pos = int(weekly_positions[-1])
            if daily_pos + 1 < daily_window or weekly_pos + 1 < weekly_window:
                continue
            daily_slice = daily_frame.iloc[daily_pos + 1 - daily_window : daily_pos + 1]
            weekly_slice = weekly_frame.iloc[weekly_pos + 1 - weekly_window : weekly_pos + 1]
            features = extract_features_for_date(daily_frame, weekly_frame, event.TradeDate, daily_window, weekly_window)
            if not features:
                continue
            all_features.update(features)
            samples.append(
                KlineWindowSample(
                    scode=event.SCode,
                    trade_date=event.TradeDate,
                    label=label,
                    gain_rate=float(event.GainRate) if hasattr(event, "GainRate") and pd.notna(event.GainRate) else None,
                    features=tuple(sorted(features)),
                    daily_55=_compact_window_to_matrix(daily_slice),
                    weekly_55=_compact_window_to_matrix(weekly_slice),
                )
            )
            batch_samples += 1
        print(f"collect {label} batch {batch_index} symbols={len(batch)} events={len(batch_events)} samples={batch_samples}", flush=True)
    return samples, all_features


def parse_date_arg(value: str | None) -> date | None:
    return parse_date(value) if value else None
