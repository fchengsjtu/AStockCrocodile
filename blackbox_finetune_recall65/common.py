from __future__ import annotations

from bisect import bisect_right
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

from blackbox_finetune.common import *  # noqa: F401,F403
from blackbox_finetune.common import DEFAULT_BASE_MODEL, DEFAULT_STAT_TYPE, DEFAULT_WINDOW, SampleEvent
from llm_finetune.common import compact_date, iter_batches, load_kline_map, parse_date

DEFAULT_OUTPUT_DIR_SHORT = Path("blackbox_finetune_recall65") / "runs" / "qwen2.5-0.5b-blackbox-recall65-short-lora"
DEFAULT_OUTPUT_DIR_LONG = Path("blackbox_finetune_recall65") / "runs" / "qwen2.5-0.5b-blackbox-recall65-long-lora"
DEFAULT_OUTPUT_DIR_XLONG = Path("blackbox_finetune_recall65") / "runs" / "qwen2.5-0.5b-blackbox-recall65-xlong-lora"
DEFAULT_OUTPUT_DIR_XXLONG = Path("blackbox_finetune_recall65") / "runs" / "qwen2.5-0.5b-blackbox-recall65-xxlong-lora"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR_LONG
DEFAULT_TRAIN_START_DATE = "20200101"
DEFAULT_TRAIN_END_DATE = "20251231"
DEFAULT_VALIDATION_START_DATE = "20260101"
DEFAULT_VALIDATION_END_DATE = "20260430"
DEFAULT_MIN_POSITIVE_RECALL = 0.65
DEFAULT_TRAIN_SEED = 20260565
CSV_COLUMNS = "dt/o/h/l/c/v/a/m5/m13/m34/m55"
SYSTEM_PROMPT = "Classify A-share surge. Return JSON."
SHORT_SAMPLE_MODE = "short"
LONG_SAMPLE_MODE = "long"
XLONG_SAMPLE_MODE = "xlong"
XXLONG_SAMPLE_MODE = "xxlong"
DEFAULT_SAMPLE_MODE = LONG_SAMPLE_MODE
SAMPLE_MODES = {
    SHORT_SAMPLE_MODE: {"daily": 8, "weekly": 5, "monthly": 0, "max_seq_length": 1024},
    LONG_SAMPLE_MODE: {"daily": 13, "weekly": 8, "monthly": 5, "max_seq_length": 2048},
    XLONG_SAMPLE_MODE: {"daily": 21, "weekly": 13, "monthly": 8, "max_seq_length": 3072},
    XXLONG_SAMPLE_MODE: {"daily": 34, "weekly": 21, "monthly": 13, "max_seq_length": 4096},
}
COMPACT_DAILY_WINDOW = SAMPLE_MODES[DEFAULT_SAMPLE_MODE]["daily"]
COMPACT_WEEKLY_WINDOW = SAMPLE_MODES[DEFAULT_SAMPLE_MODE]["weekly"]
COMPACT_MONTHLY_WINDOW = SAMPLE_MODES[DEFAULT_SAMPLE_MODE]["monthly"]
DEFAULT_MAX_SEQ_LENGTH = SAMPLE_MODES[DEFAULT_SAMPLE_MODE]["max_seq_length"]
USE_PARTIAL_WEEKLY_BAR = False


def normalize_sample_mode(sample_mode: str | None) -> str:
    mode = (sample_mode or DEFAULT_SAMPLE_MODE).lower().strip()
    if mode not in SAMPLE_MODES:
        raise ValueError(f"unsupported sample mode: {sample_mode}; expected one of {sorted(SAMPLE_MODES)}")
    return mode


def sample_mode_config(sample_mode: str | None) -> dict[str, int]:
    return SAMPLE_MODES[normalize_sample_mode(sample_mode)]


def default_max_seq_length(sample_mode: str | None = None) -> int:
    return sample_mode_config(sample_mode)["max_seq_length"]

def default_data_dir(sample_mode: str | None = None) -> Path:
    mode = normalize_sample_mode(sample_mode or os.environ.get("SAMPLE_MODE"))
    return Path("blackbox_finetune_recall65") / f"data_no_partial_week_{mode}"


def default_validation_dir(sample_mode: str | None = None) -> Path:
    mode = normalize_sample_mode(sample_mode or os.environ.get("SAMPLE_MODE"))
    return Path("blackbox_finetune_recall65") / f"data_evaluation_no_partial_week_{mode}"


DEFAULT_DATA_DIR = default_data_dir(os.environ.get("SAMPLE_MODE"))
DEFAULT_VALIDATION_DIR = default_validation_dir(os.environ.get("SAMPLE_MODE"))

def default_output_dir(sample_mode: str | None = None) -> Path:
    mode = normalize_sample_mode(sample_mode or os.environ.get("SAMPLE_MODE"))
    if mode == SHORT_SAMPLE_MODE:
        return DEFAULT_OUTPUT_DIR_SHORT
    if mode == XLONG_SAMPLE_MODE:
        return DEFAULT_OUTPUT_DIR_XLONG
    if mode == XXLONG_SAMPLE_MODE:
        return DEFAULT_OUTPUT_DIR_XXLONG
    return DEFAULT_OUTPUT_DIR_LONG



def label_answer(label: int) -> str:
    return '{"p":1}' if label else '{"p":0}'


def _row_date(row: dict) -> date:
    return parse_date(row["date"])


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
            else:
                values.append(_ratio_number(row.get(key), close_avg))
        lines.append(",".join(values))
    return "\n".join(lines)


def _has_positive_ma13(rows: list[dict]) -> bool:
    return all(_float_value(row.get("ma13")) > 0 for row in rows)


def build_compact_prompt(
    scode: str,
    anchor_date: date,
    daily_rows: list[dict],
    weekly_rows: list[dict],
    monthly_rows: list[dict] | None = None,
    sample_mode: str | None = None,
    daily_window: int | None = None,
    weekly_window: int | None = None,
    monthly_window: int | None = None,
) -> str:
    mode = normalize_sample_mode(sample_mode)
    config = sample_mode_config(mode)
    daily_count = daily_window or config["daily"]
    weekly_count = weekly_window or config["weekly"]
    monthly_count = config["monthly"] if monthly_window is None else monthly_window
    daily = daily_rows[-daily_count:] if daily_count > 0 else []
    weekly = weekly_rows[-weekly_count:] if weekly_count > 0 else []
    monthly = (monthly_rows or [])[-monthly_count:] if monthly_count > 0 else []
    parts = [
        f"s={scode}",
        f"mode={mode}",
        f"cols={CSV_COLUMNS}",
        "D",
        _compact_kline_csv(daily),
        "W",
        _compact_kline_csv(weekly),
    ]
    if monthly_count > 0:
        parts.extend(["M", _compact_kline_csv(monthly)])
    return "\n".join(parts)


def build_messages(
    scode: str,
    anchor_date: date,
    daily_rows: list[dict],
    weekly_rows: list[dict],
    monthly_rows: list[dict] | None = None,
    label: int | None = None,
    sample_mode: str | None = None,
) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_compact_prompt(scode, anchor_date, daily_rows, weekly_rows, monthly_rows, sample_mode)},
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
        return build_messages(
            str(data["scode"]),
            parse_date(data["anchor_date"]),
            _rows_from_compact_payload(data["daily_55"]),
            _rows_from_compact_payload(data["weekly_55"]),
            _rows_from_compact_payload(data.get("monthly_55", [])),
            int(row["metadata"]["label"]) if len(messages) > 2 else None,
            DEFAULT_SAMPLE_MODE,
        )
    except Exception:
        return messages


def _average_close(rows: list[dict], count: int) -> float:
    if len(rows) < count:
        return 0.0
    return sum(_float_value(row.get("close")) for row in rows[-count:]) / count


def _week_start(value: date) -> date:
    return value - timedelta(days=value.weekday())


def _build_daily_week_map(daily_rows: list[dict]) -> dict[date, list[dict]]:
    daily_week_map: dict[date, list[dict]] = {}
    for row in daily_rows:
        row_date = _row_date(row)
        daily_week_map.setdefault(_week_start(row_date), []).append(row)
    return daily_week_map


def _pick_window_by_dates(rows: list[dict], row_dates: list[str], anchor_date: date, window: int) -> list[dict] | None:
    if window <= 0:
        return []
    index = bisect_right(row_dates, compact_date(anchor_date))
    if index < window:
        return None
    return rows[index - window : index]


def build_partial_weekly_bar(
    daily_rows: list[dict],
    anchor_date: date,
    daily_week_map: dict[date, list[dict]] | None = None,
) -> dict | None:
    if not USE_PARTIAL_WEEKLY_BAR or anchor_date.weekday() > 3:
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
        "open": _float_value(week_rows[0].get("open")),
        "high": max(_float_value(row.get("high")) for row in week_rows),
        "low": min(_float_value(row.get("low")) for row in week_rows),
        "close": _float_value(week_rows[-1].get("close")),
        "volume": sum(_float_value(row.get("volume")) for row in week_rows),
        "amount": sum(_float_value(row.get("amount")) for row in week_rows),
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


def pick_monthly_window(monthly_rows: list[dict], anchor_date: date, window: int, monthly_dates: list[str] | None = None) -> list[dict] | None:
    if window <= 0:
        return []
    if monthly_dates is None:
        monthly_dates = [row["date"] for row in monthly_rows]
    return _pick_window_by_dates(monthly_rows, monthly_dates, anchor_date, window)


def sample_bottom_band_ratio() -> float:
    raw_value = os.environ.get("SAMPLE_BOTTOM_BAND_RATIO", "0.10")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = 0.10
    return min(max(value, 0.0), 1.0)


def _is_close_in_bottom_band(daily: list[dict] | None, range_rows: list[dict] | None, bottom_ratio: float | None = None) -> bool:
    if not daily or not range_rows:
        return False
    ratio = sample_bottom_band_ratio() if bottom_ratio is None else min(max(bottom_ratio, 0.0), 1.0)
    current_close = _float_value(daily[-1].get("close"))
    lows = [_float_value(row.get("low")) for row in range_rows]
    highs = [_float_value(row.get("high")) for row in range_rows]
    low = min(lows) if lows else 0.0
    high = max(highs) if highs else 0.0
    if current_close <= 0 or low <= 0 or high <= 0:
        return False
    if high <= low:
        return current_close <= low
    bottom_threshold = low + (high - low) * ratio
    return current_close <= bottom_threshold


def _sample_windows_are_valid(sample_mode: str, weekly: list[dict], monthly: list[dict] | None, daily: list[dict] | None = None) -> bool:
    if sample_mode == SHORT_SAMPLE_MODE:
        return len(weekly) >= 5 and _has_positive_ma13(weekly[-5:]) and _is_close_in_bottom_band(daily, weekly)
    config = sample_mode_config(sample_mode)
    monthly_required = config.get("monthly", 0)
    if monthly_required > 0 and (monthly is None or len(monthly) < monthly_required):
        return False
    if sample_mode in {XLONG_SAMPLE_MODE, XXLONG_SAMPLE_MODE}:
        return _is_close_in_bottom_band(daily, monthly)
    return _is_close_in_bottom_band(daily, weekly)

DELISTED_NAME_MARKERS = ("\u9000\u5e02",)
DELISTED_NAME_PREFIXES = ("\u9000", "PT")
DELISTED_NAME_SUFFIXES = ("\u9000",)


def _looks_delisted_stock_name(name) -> bool:
    text = str(name or "").strip().upper()
    if not text:
        return False
    return (
        any(marker in text for marker in DELISTED_NAME_MARKERS)
        or any(text.startswith(prefix) for prefix in DELISTED_NAME_PREFIXES)
        or any(text.endswith(suffix) for suffix in DELISTED_NAME_SUFFIXES)
    )

def _recent_market_cutoff_date(conn, anchor_date: date, max_missing_trading_days: int) -> date | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT DATE(KTime)
            FROM dkandles
            WHERE KType = 'D' AND KTime <= %s
            ORDER BY DATE(KTime) DESC
            LIMIT %s
            """,
            (anchor_date + timedelta(days=1), max_missing_trading_days + 1),
        )
        rows = cur.fetchall()
    if len(rows) <= max_missing_trading_days:
        return None
    return parse_date(rows[-1][0])


def load_abnormal_symbols(conn, symbols: Sequence[str], anchor_date: date, max_missing_trading_days: int = 10) -> set[str]:
    if not symbols:
        return set()
    abnormal: set[str] = set()
    cutoff_date = _recent_market_cutoff_date(conn, anchor_date, max_missing_trading_days)
    for batch in iter_batches(list(symbols), 500):
        placeholders = ",".join(["%s"] * len(batch))
        with conn.cursor() as cur:
            cur.execute(f"SELECT SCode, SName FROM stockinfo WHERE SCode IN ({placeholders})", batch)
            for scode, sname in cur.fetchall():
                if _looks_delisted_stock_name(sname):
                    abnormal.add(str(scode))
        if cutoff_date is not None:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT SCode, MAX(DATE(KTime))
                    FROM dkandles
                    WHERE KType = 'D' AND SCode IN ({placeholders}) AND KTime <= %s
                    GROUP BY SCode
                    """,
                    [*batch, anchor_date + timedelta(days=1)],
                )
                latest_dates = {str(scode): parse_date(latest_date) for scode, latest_date in cur.fetchall() if latest_date is not None}
            for scode in batch:
                latest_date = latest_dates.get(str(scode))
                if latest_date is None or latest_date < cutoff_date:
                    abnormal.add(str(scode))
    return abnormal


def materialize_events(
    conn,
    events: Sequence[SampleEvent],
    daily_window: int | None,
    weekly_window: int | None,
    batch_size: int,
    sample_mode: str | None = None,
    monthly_window: int | None = None,
) -> list[dict]:
    if not events:
        return []
    mode = normalize_sample_mode(sample_mode)
    config = sample_mode_config(mode)
    daily_count = daily_window or config["daily"]
    weekly_count = weekly_window or config["weekly"]
    monthly_count = config["monthly"] if monthly_window is None else monthly_window
    start_date = min(event.anchor_date for event in events)
    end_date = max(event.anchor_date for event in events)
    lookback_start = start_date - timedelta(days=max(750, monthly_count * 45, weekly_count * 14, daily_count * 5))
    samples: list[dict] = []
    symbols = sorted({event.scode for event in events})
    by_symbol: dict[str, list[SampleEvent]] = {}
    for event in events:
        by_symbol.setdefault(event.scode, []).append(event)
    for batch_index, batch in enumerate(iter_batches(symbols, batch_size), start=1):
        daily_map = load_kline_map(conn, "dkandles", "D", batch, lookback_start, end_date)
        weekly_map = load_kline_map(conn, "wkandles", "W", batch, lookback_start, end_date)
        monthly_map = load_kline_map(conn, "mkandles", "M", batch, lookback_start, end_date) if monthly_count > 0 else {}
        batch_count = 0
        for scode in batch:
            daily_rows = daily_map.get(scode, [])
            weekly_rows = weekly_map.get(scode, [])
            monthly_rows_all = monthly_map.get(scode, [])
            daily_dates = [row["date"] for row in daily_rows]
            weekly_dates = [row["date"] for row in weekly_rows]
            monthly_dates = [row["date"] for row in monthly_rows_all]
            daily_week_map = _build_daily_week_map(daily_rows) if USE_PARTIAL_WEEKLY_BAR else None
            for event in by_symbol.get(scode, []):
                daily = _pick_window_by_dates(daily_rows, daily_dates, event.anchor_date, daily_count)
                weekly = pick_weekly_window(
                    weekly_rows,
                    daily_rows,
                    event.anchor_date,
                    weekly_count,
                    weekly_dates=weekly_dates,
                    daily_week_map=daily_week_map,
                )
                monthly = pick_monthly_window(monthly_rows_all, event.anchor_date, monthly_count, monthly_dates) if monthly_count > 0 else []
                if daily is None or weekly is None or monthly is None:
                    continue
                if not _sample_windows_are_valid(mode, weekly, monthly, daily):
                    continue
                samples.append(
                    {
                        "messages": build_messages(scode, event.anchor_date, daily, weekly, monthly, event.label, mode),
                        "metadata": {
                            "scode": scode,
                            "anchor_date": event.anchor_date.isoformat(),
                            "label": event.label,
                            "source": event.source,
                            "gain_rate": event.gain_rate,
                            "sample_mode": mode,
                            "daily_window": daily_count,
                            "weekly_window": weekly_count,
                            "monthly_window": monthly_count,
                        },
                    }
                )
                batch_count += 1
        print(f"materialize batch={batch_index} symbols={len(batch)} samples={batch_count} mode={mode}", flush=True)
    return samples
