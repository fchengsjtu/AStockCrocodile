from __future__ import annotations

from bisect import bisect_right
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

from blackbox_finetune.common import *  # noqa: F401,F403
from blackbox_finetune.common import DEFAULT_BASE_MODEL, DEFAULT_STAT_TYPE, DEFAULT_WINDOW, SampleEvent
from llm_finetune.common import compact_date, iter_batches, load_kline_map, parse_date, pick_window

DEFAULT_DATA_DIR = Path("blackbox_finetune_recall60") / "data_partial_week"
DEFAULT_VALIDATION_DIR = Path("blackbox_finetune_recall60") / "data_validation_partial_week"
DEFAULT_OUTPUT_DIR = Path("blackbox_finetune_recall60") / "runs" / "qwen2.5-0.5b-blackbox-recall60-lora"
DEFAULT_TRAIN_START_DATE = "20110101"
DEFAULT_TRAIN_END_DATE = "20241231"
DEFAULT_VALIDATION_START_DATE = "20260101"
DEFAULT_VALIDATION_END_DATE = "20260430"
DEFAULT_MIN_POSITIVE_RECALL = 0.60
DEFAULT_TRAIN_SEED = 20260560
CSV_COLUMNS = "dt/o/h/l/c/v/a/m5/m13/m34/m55"
COMPACT_DAILY_WINDOW = 21
COMPACT_WEEKLY_WINDOW = 13
SYSTEM_PROMPT = "Classify A-share surge. Return JSON."


def label_answer(label: int) -> str:
    return '{"p":1}' if label else '{"p":0}'


def _scaled_number(number: float, scale: float, suffix: str, decimals: int) -> str:
    text = f"{number / scale:.{decimals}f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return f"{text}{suffix}"


def _row_date(row: dict) -> date:
    return parse_date(row["date"])


def _csv_number(value) -> str:
    if isinstance(value, str):
        return value
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return str(value)
    abs_number = abs(number)
    if abs_number >= 1_000_000_000:
        return _scaled_number(number, 1_000_000_000, "b", 2)
    if abs_number >= 1_000_000:
        return _scaled_number(number, 1_000_000, "m", 2)
    if abs_number >= 1000:
        return _scaled_number(number, 1000, "k", 1)
    return f"{number:.2f}".rstrip("0").rstrip(".") or "0"


def _float_value(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _average_positive(rows: list[dict], key: str) -> float:
    values = [_float_value(row.get(key)) for row in rows]
    positives = [value for value in values if value > 0]
    return sum(positives) / len(positives) if positives else 0.0


def _ratio_number(value, denominator: float) -> str:
    if denominator <= 0:
        return "0"
    return f"{_float_value(value) / denominator:.2f}".rstrip("0").rstrip(".") or "0"


def _compact_kline_csv(rows: list[dict]) -> str:
    keys = ["open", "high", "low", "close", "volume", "amount", "ma5", "ma13", "ma34", "ma55"]
    close_avg = _average_positive(rows, "close")
    volume_avg = _average_positive(rows, "volume")
    amount_avg = _average_positive(rows, "amount")
    lines = []
    for index, row in enumerate(rows, start=1):
        values = [str(index)]
        for key in keys:
            if key == "volume":
                values.append(_ratio_number(row.get(key), volume_avg))
            elif key == "amount":
                values.append(_ratio_number(row.get(key), amount_avg))
            elif key in {"open", "high", "low", "close", "ma5", "ma13", "ma34", "ma55"}:
                values.append(_ratio_number(row.get(key), close_avg))
            else:
                values.append(_csv_number(row.get(key)))
        lines.append(",".join(values))
    return "\n".join(lines)


def build_compact_prompt(
    scode: str,
    anchor_date: date,
    daily_55: list[dict],
    weekly_55: list[dict],
    daily_window: int = COMPACT_DAILY_WINDOW,
    weekly_window: int = COMPACT_WEEKLY_WINDOW,
) -> str:
    daily_rows = daily_55[-daily_window:] if daily_window > 0 else []
    weekly_rows = weekly_55[-weekly_window:] if weekly_window > 0 else []
    return (
        f"s={scode}\n"
        f"cols={CSV_COLUMNS}\n"
        "D\n"
        f"{_compact_kline_csv(daily_rows)}\n"
        "W\n"
        f"{_compact_kline_csv(weekly_rows)}"
    )


def build_messages(scode: str, anchor_date: date, daily_55: list[dict], weekly_55: list[dict], label: int | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_compact_prompt(scode, anchor_date, daily_55, weekly_55)},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": label_answer(label)})
    return messages


def _rows_from_compact_payload(rows: list[list]) -> list[dict]:
    keys = ["date", "open", "high", "low", "close", "volume", "amount", "ma5", "ma13", "ma34", "ma55"]
    return [dict(zip(keys, row)) for row in rows]


def compact_messages_from_sample(row: dict) -> list[dict]:
    messages = row["messages"]
    user_content = messages[1]["content"]
    if "daily_55" not in user_content or "weekly_55" not in user_content:
        return messages
    try:
        import json

        data = json.loads(user_content)
        sample_messages = build_messages(
            str(data["scode"]),
            parse_date(data["anchor_date"]),
            _rows_from_compact_payload(data["daily_55"]),
            _rows_from_compact_payload(data["weekly_55"]),
            int(row["metadata"]["label"]) if len(messages) > 2 else None,
        )
        return sample_messages
    except Exception:
        return messages


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
