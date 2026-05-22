from __future__ import annotations

from bisect import bisect_right
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

from blackbox_finetune.common import *  # noqa: F401,F403
from blackbox_finetune.common import DEFAULT_BASE_MODEL, DEFAULT_STAT_TYPE, DEFAULT_WINDOW, SampleEvent, build_messages
from llm_finetune.common import compact_date, iter_batches, load_kline_map, parse_date

DEFAULT_DATA_DIR = Path("blackbox_finetune_recall80") / "data_partial_week"
DEFAULT_VALIDATION_DIR = Path("blackbox_finetune_recall80") / "data_validation_partial_week"
DEFAULT_OUTPUT_DIR = Path("blackbox_finetune_recall80") / "runs" / "qwen2.5-0.5b-blackbox-recall80-lora"
DEFAULT_TRAIN_START_DATE = "20110101"
DEFAULT_TRAIN_END_DATE = "20241231"
DEFAULT_VALIDATION_START_DATE = "20260101"
DEFAULT_VALIDATION_END_DATE = "20260430"
DEFAULT_MIN_POSITIVE_RECALL = 0.80
DEFAULT_TRAIN_SEED = 20260580


def _row_date(row: dict) -> date:
    return parse_date(row["date"])


def _average_close(rows: list[dict], count: int) -> float:
    if len(rows) < count:
        return 0.0
    return sum(float(row.get("close") or 0) for row in rows[-count:]) / count


def _week_start(value: date) -> date:
    return value - timedelta(days=value.weekday())


def _build_daily_week_map(daily_rows: list[dict]) -> dict[date, list[dict]]:
    daily_week_map: dict[date, list[dict]] = {}
    for row in daily_rows:
        row_date = _row_date(row)
        daily_week_map.setdefault(_week_start(row_date), []).append(row)
    return daily_week_map


def _pick_window_by_dates(rows: list[dict], row_dates: list[str], anchor_date: date, window: int) -> list[dict] | None:
    index = bisect_right(row_dates, compact_date(anchor_date))
    if index < window:
        return None
    return rows[index - window : index]


def build_partial_weekly_bar(
    daily_rows: list[dict],
    anchor_date: date,
    daily_week_map: dict[date, list[dict]] | None = None,
) -> dict | None:
    """Build an in-memory Monday-to-anchor weekly bar for Monday-Thursday anchors."""
    if anchor_date.weekday() > 3:
        return None
    week_start = _week_start(anchor_date)
    if daily_week_map is None:
        week_rows = [row for row in daily_rows if week_start <= _row_date(row) <= anchor_date]
    else:
        anchor_compact = compact_date(anchor_date)
        week_rows = [row for row in daily_week_map.get(week_start, []) if row["date"] <= anchor_compact]
    if not week_rows:
        return None
    week_rows = sorted(week_rows, key=_row_date)
    return {
        "date": compact_date(anchor_date),
        "open": float(week_rows[0].get("open") or 0),
        "high": max(float(row.get("high") or 0) for row in week_rows),
        "low": min(float(row.get("low") or 0) for row in week_rows),
        "close": float(week_rows[-1].get("close") or 0),
        "volume": sum(float(row.get("volume") or 0) for row in week_rows),
        "amount": sum(float(row.get("amount") or 0) for row in week_rows),
        "ma5": 0.0,
        "ma13": 0.0,
        "ma34": 0.0,
        "ma55": 0.0,
    }


def pick_weekly_window(
    weekly_rows: list[dict],
    daily_rows: list[dict],
    anchor_date: date,
    window: int,
    weekly_dates: list[str] | None = None,
    daily_week_map: dict[date, list[dict]] | None = None,
) -> list[dict] | None:
    if weekly_dates is None:
        weekly_dates = [row["date"] for row in weekly_rows]
    index = bisect_right(weekly_dates, compact_date(anchor_date))
    weekly = weekly_rows[max(0, index - max(window, 55)) : index]
    partial = build_partial_weekly_bar(daily_rows, anchor_date, daily_week_map)
    if partial is not None:
        partial_date = _row_date(partial)
        weekly = [row for row in weekly if _row_date(row) < partial_date]
        weekly.append(partial)
        weekly = sorted(weekly, key=_row_date)
        for count, key in ((5, "ma5"), (13, "ma13"), (34, "ma34"), (55, "ma55")):
            weekly[-1][key] = _average_close(weekly, count)
    if len(weekly) < window:
        return None
    return weekly[-window:]


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
            daily_rows = daily_map.get(scode, [])
            weekly_rows = weekly_map.get(scode, [])
            daily_dates = [row["date"] for row in daily_rows]
            weekly_dates = [row["date"] for row in weekly_rows]
            daily_week_map = _build_daily_week_map(daily_rows)
            for event in by_symbol.get(scode, []):
                daily = _pick_window_by_dates(daily_rows, daily_dates, event.anchor_date, daily_window)
                weekly = pick_weekly_window(
                    weekly_rows,
                    daily_rows,
                    event.anchor_date,
                    weekly_window,
                    weekly_dates=weekly_dates,
                    daily_week_map=daily_week_map,
                )
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
                            "weekly_partial": event.anchor_date.weekday() <= 3,
                        },
                    }
                )
                batch_count += 1
        print(f"materialize batch={batch_index} symbols={len(batch)} samples={batch_count}", flush=True)
    return samples
