from __future__ import annotations

import json
import os
import re
import socket
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pymysql

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA_DIR = Path("llm_finetune") / "data"
DEFAULT_OUTPUT_DIR = Path("llm_finetune") / "runs" / "qwen2.5-0.5b-stock-lora"
DEFAULT_WINDOW = 55
DEFAULT_STAT_TYPE = "short_term_surge_3d_20pct"
DEFAULT_MIN_SUCCESS_RATE = 0.20
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = (
    "You are an A-share technical-pattern classifier. "
    "Given exactly 55 daily K-lines and up to 55 weekly K-lines ending at anchor_date, "
    "decide whether the setup matches a reusable short-term surge selection pattern. "
    "Return strict JSON only."
)


@dataclass(frozen=True)
class Event:
    scode: str
    anchor_date: date
    label: int
    gain_rate: float | None = None


def parse_date(value: str | date | datetime) -> date:
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
    raise ValueError(f"invalid date: {value!r}; expected YYYYMMDD or YYYY-MM-DD")


def compact_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def load_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    pattern = re.compile(r"(?:\$env:)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*['\"]?([^'\";\r\n]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        for key, value in pattern.findall(line):
            values[key] = value.strip()
    return values


def load_env() -> dict[str, str]:
    values = {}
    values.update(load_key_value_file(PROJECT_ROOT / "env.txt"))
    values.update(load_key_value_file(Path(__file__).resolve().parent / "config.env"))
    for key in (
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "MYSQL_DATABASE",
        "MYSQL_CHARSET",
        "HF_ENDPOINT",
        "BASE_MODEL",
    ):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def detect_wsl_windows_host() -> str | None:
    try:
        for line in Path("/etc/resolv.conf").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("nameserver "):
                return line.split()[1]
    except OSError:
        return None
    return None


def resolve_mysql_host(host: str) -> str:
    if host in {"127.0.0.1", "localhost", "::1"} and is_wsl():
        return os.environ.get("WSL_MYSQL_HOST") or detect_wsl_windows_host() or host
    return host


def mysql_connect():
    env = load_env()
    host = resolve_mysql_host(env.get("MYSQL_HOST", "127.0.0.1"))
    try:
        return pymysql.connect(
            host=host,
            port=int(env.get("MYSQL_PORT", "3306")),
            user=env.get("MYSQL_USER", "root"),
            password=env.get("MYSQL_PASSWORD", ""),
            database=env.get("MYSQL_DATABASE", "emstocks"),
            charset=env.get("MYSQL_CHARSET", "utf8mb4"),
            autocommit=False,
            connect_timeout=20,
            read_timeout=120,
            write_timeout=120,
        )
    except RuntimeError as exc:
        if "cryptography" in str(exc):
            raise RuntimeError(
                "MySQL uses caching_sha2_password/sha256_password; install cryptography:\n"
                "  python -m pip install cryptography\n"
                "or rerun the one-click script."
            ) from exc
        raise
    except pymysql.err.OperationalError as exc:
        if exc.args and exc.args[0] == 1130:
            raise RuntimeError(
                f"MySQL rejected host {socket.gethostname()} / {host}. Grant this client, for example:\n"
                "  CREATE USER IF NOT EXISTS 'fcheng'@'172.%' IDENTIFIED BY '123456';\n"
                "  GRANT ALL PRIVILEGES ON emstocks.* TO 'fcheng'@'172.%';\n"
                "  FLUSH PRIVILEGES;"
            ) from exc
        raise


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_batches(items: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json_dumps(row) + "\n")
            count += 1
    return count


def kline_query(table: str, ktype: str, scodes: Sequence[str], start_date: date, end_date: date) -> tuple[str, list]:
    if table not in {"dkandles", "wkandles"}:
        raise ValueError(f"unsupported table: {table}")
    placeholders = ",".join(["%s"] * len(scodes))
    sql = f"""
        SELECT SCode, DATE(KTime) AS TradeDate, Open, High, Low, Close, Volume, Amount, MA5, MA13, MA34, MA55
        FROM {table}
        WHERE KType = %s
          AND SCode IN ({placeholders})
          AND KTime >= %s
          AND KTime < %s
        ORDER BY SCode, KTime
    """
    return sql, [ktype, *scodes, start_date, end_date + timedelta(days=1)]


def load_kline_map(conn, table: str, ktype: str, scodes: Sequence[str], start_date: date, end_date: date) -> dict[str, list[dict]]:
    if not scodes:
        return {}
    sql, params = kline_query(table, ktype, scodes, start_date, end_date)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    result: dict[str, list[dict]] = {}
    for row in rows:
        scode = str(row[0])
        result.setdefault(scode, []).append(
            {
                "date": compact_date(parse_date(row[1])),
                "open": float(row[2] or 0),
                "high": float(row[3] or 0),
                "low": float(row[4] or 0),
                "close": float(row[5] or 0),
                "volume": float(row[6] or 0),
                "amount": float(row[7] or 0),
                "ma5": float(row[8] or 0),
                "ma13": float(row[9] or 0),
                "ma34": float(row[10] or 0),
                "ma55": float(row[11] or 0),
            }
        )
    return result


def pick_window(rows: list[dict], anchor_date: date, window: int) -> list[dict] | None:
    anchor = compact_date(anchor_date)
    eligible = [row for row in rows if row["date"] <= anchor]
    if len(eligible) < window:
        return None
    return eligible[-window:]


def make_prompt_payload(scode: str, anchor_date: date, daily_55: list[dict], weekly_55: list[dict]) -> dict:
    return {
        "task": "classify_stock_surge_setup",
        "scode": scode,
        "anchor_date": compact_date(anchor_date),
        "input_definition": "daily_55 and weekly_55 end at or before anchor_date; use only these historical K-lines.",
        "columns": ["date", "open", "high", "low", "close", "volume", "amount", "ma5", "ma13", "ma34", "ma55"],
        "daily_55": compact_kline_rows(daily_55),
        "weekly_55": compact_kline_rows(weekly_55),
        "output_schema": {
            "label": "positive or negative",
            "success_probability": "0.0 to 1.0",
            "reason": "short technical explanation",
        },
    }


def build_messages(scode: str, anchor_date: date, daily_55: list[dict], weekly_55: list[dict], label: int | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json_dumps(make_prompt_payload(scode, anchor_date, daily_55, weekly_55))},
    ]
    if label is not None:
        response = {
            "label": "positive" if label else "negative",
            "success_probability": 0.8 if label else 0.1,
            "reason": "Historical label from klinestatistics." if label else "No matching surge label in klinestatistics.",
        }
        messages.append({"role": "assistant", "content": json_dumps(response)})
    return messages


def compact_kline_rows(rows: list[dict]) -> list[list]:
    keys = ["date", "open", "high", "low", "close", "volume", "amount", "ma5", "ma13", "ma34", "ma55"]
    compact = []
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key)
            if isinstance(value, float):
                values.append(round(value, 4))
            else:
                values.append(value)
        compact.append(values)
    return compact
