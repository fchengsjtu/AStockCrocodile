from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_finetune.common import (
    BASE_MODEL,
    compact_date,
    compact_kline_rows,
    iter_batches,
    json_dumps,
    load_kline_map,
    mysql_connect,
    parse_date,
    pick_window,
    read_jsonl,
    write_jsonl,
)

DEFAULT_DATA_DIR = Path("blackbox_finetune") / "data"
DEFAULT_OUTPUT_DIR = Path("blackbox_finetune") / "runs" / "qwen2.5-0.5b-blackbox-lora"
DEFAULT_WINDOW = 55
DEFAULT_STAT_TYPE = "short_term_surge_3d_20pct"
DEFAULT_BASE_MODEL = BASE_MODEL
SYSTEM_PROMPT = (
    "You are a black-box A-share surge classifier. "
    "Use only the supplied 55 daily K-lines and 55 weekly K-lines ending at the anchor date. "
    "Return strict JSON only."
)


@dataclass(frozen=True)
class SampleEvent:
    scode: str
    anchor_date: date
    label: int
    source: str
    gain_rate: float | None = None


def event_key(scode: str, anchor_date: date) -> tuple[str, date]:
    return str(scode), anchor_date


def label_answer(label: int) -> str:
    return json_dumps(
        {
            "label": "positive" if label else "negative",
            "positive_probability": 0.85 if label else 0.15,
        }
    )


def make_payload(scode: str, anchor_date: date, daily_55: list[dict], weekly_55: list[dict]) -> dict:
    return {
        "task": "blackbox_stock_surge_classification",
        "scode": scode,
        "anchor_date": compact_date(anchor_date),
        "input_rule": "Use only K-lines on or before anchor_date. Do not use future data.",
        "columns": ["date", "open", "high", "low", "close", "volume", "amount", "ma5", "ma13", "ma34", "ma55"],
        "daily_55": compact_kline_rows(daily_55),
        "weekly_55": compact_kline_rows(weekly_55),
        "output_schema": {"label": "positive|negative", "positive_probability": "0.0-1.0"},
    }


def build_messages(scode: str, anchor_date: date, daily_55: list[dict], weekly_55: list[dict], label: int | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json_dumps(make_payload(scode, anchor_date, daily_55, weekly_55))},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": label_answer(label)})
    return messages


def materialize_events(
    conn,
    events: Sequence[SampleEvent],
    daily_window: int,
    weekly_window: int,
    batch_size: int,
) -> list[dict]:
    if not events:
        return []
    start_date = min(event.anchor_date for event in events)
    end_date = max(event.anchor_date for event in events)
    lookback_start = start_date - timedelta(days=max(500, weekly_window * 10, daily_window * 4))
    samples: list[dict] = []
    symbols = sorted({event.scode for event in events})
    by_symbol: dict[str, list[SampleEvent]] = {}
    for event in events:
        by_symbol.setdefault(event.scode, []).append(event)
    for batch_index, batch in enumerate(iter_batches(symbols, batch_size), start=1):
        daily_map = load_kline_map(conn, "dkandles", "D", batch, lookback_start, end_date)
        weekly_map = load_kline_map(conn, "wkandles", "W", batch, lookback_start, end_date)
        batch_count = 0
        for scode in batch:
            for event in by_symbol.get(scode, []):
                daily = pick_window(daily_map.get(scode, []), event.anchor_date, daily_window)
                weekly = pick_window(weekly_map.get(scode, []), event.anchor_date, weekly_window)
                if daily is None or weekly is None:
                    continue
                samples.append(
                    {
                        "messages": build_messages(scode, event.anchor_date, daily, weekly, event.label),
                        "metadata": {
                            "scode": scode,
                            "anchor_date": event.anchor_date.isoformat(),
                            "label": event.label,
                            "source": event.source,
                            "gain_rate": event.gain_rate,
                        },
                    }
                )
                batch_count += 1
        print(f"materialize batch={batch_index} symbols={len(batch)} samples={batch_count}", flush=True)
    return samples

