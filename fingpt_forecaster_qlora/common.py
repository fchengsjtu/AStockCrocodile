from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import pymysql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = Path("fingpt_forecaster_qlora") / "data"
DEFAULT_OUTPUT_DIR = Path("fingpt_forecaster_qlora") / "runs" / "astock-fingpt-forecaster-qlora"
DEFAULT_BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf"
DEFAULT_FINGPT_FORECASTER_ADAPTER = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
DEFAULT_STAT_TYPE = "short_term_surge_3d_20pct"
SYSTEM_PROMPT = (
    "You are FinGPT-Forecaster adapted for China A-share short-term opportunity forecasting. "
    "Given recent daily/weekly OHLCV data and nearby market news, forecast whether the stock "
    "matches a reusable bullish setup. Return strict JSON only."
)


@dataclass(frozen=True)
class ForecastSample:
    scode: str
    trade_date: date
    label: int
    gain_rate: float
    daily_rows: list[dict]
    weekly_rows: list[dict]
    news_rows: list[dict]


def load_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    assignments = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assignments.extend(part.strip() for part in line.split(";") if part.strip())
    for text in assignments:
        if "=" not in text:
            continue
        if text.lower().startswith("$env:"):
            text = text[5:]
        key, value = text.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        values[key] = value.strip().strip("'").strip('"')
    return values


def load_env(config_path: Path | None = None) -> dict[str, str]:
    values = load_key_value_file(PROJECT_ROOT / "env.txt")
    if config_path is None:
        config_path = PROJECT_ROOT / "fingpt_forecaster_qlora" / "config.env"
    values.update(load_key_value_file(config_path))
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return {**values, **os.environ}


def mysql_connect():
    load_env()
    return pymysql.connect(
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", ""),
        database=os.environ.get("MYSQL_DATABASE", "emstocks"),
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=20,
        read_timeout=120,
        write_timeout=120,
    )


def parse_date(value: str | date | datetime | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid date: {value}")
    return parsed.date()


def json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def compact_kline_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    result = df.copy()
    result["KDate"] = pd.to_datetime(result["KTime"]).dt.strftime("%Y-%m-%d")
    columns = ["KDate", "Open", "Close", "High", "Low", "Volume", "Amount", "MA5", "MA13", "MA34", "MA55"]
    rows = []
    for row in result[columns].itertuples(index=False):
        item = row._asdict()
        for key, value in list(item.items()):
            if isinstance(value, float):
                item[key] = round(value, 6)
            elif pd.isna(value):
                item[key] = None
        rows.append(item)
    return rows


def load_kline_window(conn, table: str, ktype: str, scode: str, trade_date: date, window: int) -> pd.DataFrame:
    sql = f"""
        SELECT KTime, Open, Close, High, Low, Volume, Amount, MA5, MA13, MA34, MA55
        FROM {table}
        WHERE SCode = %s AND KType = %s AND DATE(KTime) <= %s
        ORDER BY KTime DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (scode, ktype, trade_date, window))
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["KTime", "Open", "Close", "High", "Low", "Volume", "Amount", "MA5", "MA13", "MA34", "MA55"])
    if df.empty:
        return df
    for column in ["Open", "Close", "High", "Low", "Volume", "Amount", "MA5", "MA13", "MA34", "MA55"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values("KTime").reset_index(drop=True)


def load_nearby_news(conn, trade_date: date, days: int = 3, limit: int = 12) -> list[dict]:
    sql = """
        SELECT Title, SourceName, PublishTime, CredibilityLevel, Heat, RelatedConcepts, ConceptHeat
        FROM news
        WHERE PublishTime >= %s AND PublishTime < %s
        ORDER BY PublishTime DESC, Heat DESC
        LIMIT %s
    """
    start = trade_date - timedelta(days=days)
    end = trade_date + timedelta(days=days + 1)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (start, end, limit))
            rows = cur.fetchall()
    except Exception:
        return []
    news = []
    for title, source, publish_time, credibility, heat, related, concept_heat in rows:
        news.append(
            {
                "title": title,
                "source": source,
                "publish_time": str(publish_time) if publish_time else None,
                "credibility": int(credibility or 5),
                "heat": int(heat or 0),
                "related_concepts": related,
                "concept_heat": concept_heat,
            }
        )
    return news


def build_prompt(sample: ForecastSample) -> str:
    payload = {
        "task": "forecast_short_term_surge_setup",
        "stock_code": sample.scode,
        "anchor_date": str(sample.trade_date),
        "daily_ohlcv_55": sample.daily_rows,
        "weekly_ohlcv_55": sample.weekly_rows,
        "nearby_news": sample.news_rows,
        "output_schema": {
            "label": "positive|negative",
            "success_probability": "0.0-1.0",
            "confidence": "0.0-1.0",
            "key_patterns": ["3 to 8 concise reusable pattern clauses"],
            "risk_factors": ["concise risk clauses"],
        },
    }
    return json_dumps(payload)


def build_answer(sample: ForecastSample, min_success_rate: float) -> str:
    positive = sample.label == 1
    probability = max(min_success_rate, 0.5) if positive else min(0.2, max(0.0, min_success_rate - 0.15))
    return json_dumps(
        {
            "label": "positive" if positive else "negative",
            "success_probability": round(float(probability), 4),
            "confidence": 0.65 if positive else 0.55,
            "key_patterns": [
                "recent daily price action precedes a short-term surge sample",
                "weekly context is included to avoid single-day noise",
                "volume and moving-average structure are evaluated together",
            ]
            if positive
            else [],
            "risk_factors": ["historical setup did not match the surge label"] if not positive else [],
        }
    )


def sample_to_messages(sample: ForecastSample, min_success_rate: float) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(sample)},
            {"role": "assistant", "content": build_answer(sample, min_success_rate)},
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
