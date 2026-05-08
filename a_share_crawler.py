from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import akshare as ak
import pandas as pd
import pymysql
import requests.sessions
from apscheduler.schedulers.blocking import BlockingScheduler
from tqdm import tqdm

RUN_PERIODS = {"daily"}
GENERATE_PERIODS = {"weekly", "monthly", "all"}
MODES = {"full", "incremental"}
DEFAULT_START_DATE = "20100101"
DEFAULT_END_DATE = date.today().strftime("%Y%m%d")
DEFAULT_ADJUST = ""
DEFAULT_KTYPE = "D"
BATCH_SIZE = 5000
MA_WINDOWS = (5, 8, 13, 34, 55)
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
ENV_FILE = PROJECT_ROOT / "env.txt"
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
DEFAULT_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json,text/plain,*/*",
}
ORIGINAL_MERGE_ENVIRONMENT_SETTINGS = requests.sessions.Session.merge_environment_settings
ORIGINAL_SESSION_REQUEST = requests.sessions.Session.request


@dataclass(frozen=True)
class CrawlConfig:
    start_date: str
    end_date: str
    adjust: str
    sleep_seconds: float
    retries: int
    ktype: str


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"crawler-{date.today():%Y%m%d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.lower().startswith("$env:"):
        line = line[5:]
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().rstrip(";").strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    if not key:
        return None
    return key, value


def load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        parsed = parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def mysql_connect() -> pymysql.connections.Connection:
    load_env_file()
    return pymysql.connect(
        host=env("MYSQL_HOST"),
        port=int(env("MYSQL_PORT", "3306")),
        user=env("MYSQL_USER"),
        password=env("MYSQL_PASSWORD"),
        database=env("MYSQL_DATABASE"),
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=15,
        read_timeout=int(env("MYSQL_READ_TIMEOUT", "1800")),
        write_timeout=int(env("MYSQL_WRITE_TIMEOUT", "1800")),
    )


def disable_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def force_requests_direct() -> None:
    disable_proxy_env()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        settings = ORIGINAL_MERGE_ENVIRONMENT_SETTINGS(self, url, proxies, stream, verify, cert)
        settings["proxies"] = {}
        return settings

    def request(self, method, url, **kwargs):
        headers = dict(DEFAULT_REQUEST_HEADERS)
        headers.update(kwargs.pop("headers", {}) or {})
        kwargs["headers"] = headers
        kwargs["proxies"] = {}
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 20
        self.trust_env = False
        return ORIGINAL_SESSION_REQUEST(self, method, url, **kwargs)

    requests.sessions.Session.merge_environment_settings = merge_environment_settings
    requests.sessions.Session.request = request


def restore_requests_proxy_handling() -> None:
    os.environ.pop("NO_PROXY", None)
    os.environ.pop("no_proxy", None)
    requests.sessions.Session.merge_environment_settings = ORIGINAL_MERGE_ENVIRONMENT_SETTINGS
    requests.sessions.Session.request = ORIGINAL_SESSION_REQUEST


def normalize_symbol(symbol: str) -> str:
    digits = "".join(ch for ch in str(symbol).strip() if ch.isdigit())
    return digits[-6:].zfill(6)


def normalize_stock_list(stocks: pd.DataFrame, code_col: str, name_col: str) -> pd.DataFrame:
    stocks = stocks.rename(columns={code_col: "code", name_col: "name"})
    stocks["code"] = stocks["code"].astype(str).map(normalize_symbol)
    stocks["name"] = stocks["name"].astype(str).str.strip()
    stocks = stocks[stocks["code"].str.len() == 6]
    stocks = stocks.drop_duplicates(subset=["code"]).sort_values("code")
    return stocks[["code", "name"]]


def get_stock_list() -> pd.DataFrame:
    errors = []
    try:
        stocks = ak.stock_info_a_code_name()
        return normalize_stock_list(stocks, "code", "name")
    except Exception as exc:
        errors.append(f"stock_info_a_code_name: {exc}")
        logging.warning("stock_info_a_code_name failed; trying realtime spot list: %s", exc)

    try:
        stocks = ak.stock_zh_a_spot()
        return normalize_stock_list(stocks, "代码", "名称")
    except Exception as exc:
        errors.append(f"stock_zh_a_spot: {exc}")
        logging.error("stock_zh_a_spot fallback failed: %s", exc)

    raise RuntimeError("Unable to load A-share stock list. " + " | ".join(errors))


def tx_symbol(symbol: str) -> str:
    if symbol.startswith(("4", "8", "9")):
        return f"bj{symbol}"
    return f"sh{symbol}" if symbol.startswith(("5", "6")) else f"sz{symbol}"


def clean_yyyymmdd(value: str) -> str:
    return value.replace("-", "")


def effective_full_start_date(start_date: str) -> str:
    return max(clean_yyyymmdd(start_date), DEFAULT_START_DATE)


def yyyymmdd_from_date(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y%m%d")


def next_yyyymmdd(value: date | datetime | str | None) -> str | None:
    current = yyyymmdd_from_date(value)
    if current is None:
        return None
    return (datetime.strptime(current, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")


def upsert_stock_info(conn: pymysql.connections.Connection, stocks: pd.DataFrame) -> None:
    sql = """
        INSERT INTO stockinfo (SCode, SName, IsIndex)
        VALUES (%s, %s, 0)
        ON DUPLICATE KEY UPDATE
            SName = VALUES(SName),
            IsIndex = COALESCE(IsIndex, VALUES(IsIndex))
    """
    rows = [(row.code, row.name) for row in stocks.itertuples(index=False)]
    with conn.cursor() as cur:
        for batch in iter_batches(rows, BATCH_SIZE):
            cur.executemany(sql, batch)
    conn.commit()
    logging.info("Updated stockinfo basic data for %s stocks", len(rows))


def load_latest_update_map(conn: pymysql.connections.Connection, ktype: str) -> dict[str, datetime | None]:
    sql = """
        SELECT si.SCode, si.LatestUpdateKandle, dk.LatestKTime
        FROM stockinfo si
        LEFT JOIN (
            SELECT SCode, MAX(KTime) AS LatestKTime
            FROM dkandles
            WHERE KType = %s
            GROUP BY SCode
        ) dk ON dk.SCode = si.SCode
    """
    latest_map: dict[str, datetime | None] = {}
    stale_count = 0
    with conn.cursor() as cur:
        cur.execute(sql, (ktype,))
        for code, stockinfo_latest, daily_latest in cur.fetchall():
            symbol = str(code).zfill(6)
            if daily_latest is None:
                latest_map[symbol] = None
                if stockinfo_latest is not None:
                    stale_count += 1
                continue
            latest_map[symbol] = daily_latest
            if stockinfo_latest != daily_latest:
                stale_count += 1
    if stale_count:
        logging.warning("Found %s stockinfo.LatestUpdateKandle values inconsistent with dkandles; dkandles will be used for resume", stale_count)
    return latest_map


def sync_stock_latest_from_dkandles(conn: pymysql.connections.Connection, ktype: str) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE stockinfo SET LatestUpdateKandle = NULL")
        cur.execute(
            """
            UPDATE stockinfo si
            JOIN (
                SELECT SCode, MAX(KTime) AS LatestKTime
                FROM dkandles
                WHERE KType = %s
                GROUP BY SCode
            ) dk ON dk.SCode = si.SCode
            SET si.LatestUpdateKandle = dk.LatestKTime
            """,
            (ktype,),
        )
        updated = cur.rowcount
    conn.commit()
    logging.info("Synced stockinfo.LatestUpdateKandle from dkandles for %s stocks", updated)


def update_stock_latest(conn: pymysql.connections.Connection, symbol: str, latest: datetime) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE stockinfo SET LatestUpdateKandle = %s WHERE SCode = %s",
            (latest, symbol),
        )
    conn.commit()


def fetch_daily_from_tencent(symbol: str, cfg: CrawlConfig) -> pd.DataFrame:
    start_date = effective_full_start_date(cfg.start_date)
    end_date = clean_yyyymmdd(cfg.end_date)
    if start_date > end_date:
        return pd.DataFrame()

    for attempt in range(1, cfg.retries + 1):
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=tx_symbol(symbol),
                start_date=start_date,
                end_date=end_date,
                adjust=cfg.adjust,
                timeout=20,
            )
            break
        except Exception as exc:
            if attempt == cfg.retries:
                logging.error("daily %s Tencent fetch failed after %s attempts: %s", symbol, cfg.retries, exc)
                return pd.DataFrame()
            wait_seconds = min(30, attempt * 3)
            logging.warning("daily %s Tencent fetch failed on attempt %s; retrying in %s seconds: %s", symbol, attempt, wait_seconds, exc)
            time.sleep(wait_seconds)
    else:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(
        columns={
            "date": "KTime",
            "open": "Open",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "amount": "Volume",
        }
    )
    df["SCode"] = symbol
    df["KType"] = cfg.ktype
    df["KTime"] = pd.to_datetime(df["KTime"], errors="coerce").dt.normalize() + pd.Timedelta(hours=15)
    for column in ("Open", "Close", "High", "Low", "Volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Amount"] = 0
    df = df.dropna(subset=["KTime", "Open", "Close", "High", "Low", "Volume"])
    df = df[df["KTime"] >= pd.Timestamp("2010-01-01")]
    df = df.sort_values("KTime").drop_duplicates(subset=["SCode", "KType", "KTime"], keep="last")
    return df


def load_ma_warmup(conn: pymysql.connections.Connection, symbol: str, ktype: str, before_date: str) -> pd.DataFrame:
    before_dt = datetime.strptime(before_date, "%Y%m%d")
    sql = """
        SELECT SCode, KType, KTime, Amount, Volume, Open, Close, High, Low
        FROM dkandles
        WHERE SCode = %s AND KType = %s AND KTime < %s
        ORDER BY KTime DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, ktype, before_dt, max(MA_WINDOWS) - 1))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    warmup = pd.DataFrame(rows, columns=["SCode", "KType", "KTime", "Amount", "Volume", "Open", "Close", "High", "Low"])
    warmup["KTime"] = pd.to_datetime(warmup["KTime"], errors="coerce")
    for column in ("Amount", "Volume", "Open", "Close", "High", "Low"):
        warmup[column] = pd.to_numeric(warmup[column], errors="coerce")
    return warmup.sort_values("KTime")


def add_moving_averages(df: pd.DataFrame, warmup_df: pd.DataFrame | None = None) -> pd.DataFrame:
    source = df.copy().sort_values("KTime")
    if warmup_df is not None and not warmup_df.empty:
        source = pd.concat([warmup_df, source], ignore_index=True).sort_values("KTime")
    for window in MA_WINDOWS:
        source[f"MA{window}"] = source["Close"].rolling(window=window, min_periods=window).mean()
    result = source[source["KTime"].isin(df["KTime"])].copy()
    now = datetime.now()
    result["CreatedOn"] = now
    result["UpdatedOn"] = now
    return result


def none_if_nan(value):
    if pd.isna(value):
        return None
    return value


def rows_for_insert(df: pd.DataFrame) -> list[tuple]:
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            (
                row.SCode,
                row.KType,
                row.KTime.to_pydatetime(),
                none_if_nan(row.Amount),
                none_if_nan(row.Volume),
                none_if_nan(row.MA5),
                none_if_nan(row.MA13),
                none_if_nan(row.MA8),
                none_if_nan(row.Open),
                none_if_nan(row.Close),
                none_if_nan(row.High),
                none_if_nan(row.Low),
                none_if_nan(row.MA55),
                none_if_nan(row.MA34),
            )
        )
    return rows


def iter_batches(items: list[tuple], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def insert_rows(conn: pymysql.connections.Connection, rows: list[tuple]) -> int:
    if not rows:
        return 0
    insert_sql = """
        INSERT INTO dkandles
            (SCode, KType, KTime, Amount, Volume, MA5, MA13, MA8, Open, Close, High, Low, MA55, MA34)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    inserted = 0
    with conn.cursor() as cur:
        for batch in iter_batches(rows, BATCH_SIZE):
            cur.executemany(insert_sql, batch)
            inserted += len(batch)
    return inserted


def crawl_daily_to_mysql(
    mode: str,
    start_date: str,
    end_date: str,
    adjust: str,
    sleep_seconds: float,
    retries: int,
    ktype: str,
    workers: int,
) -> None:
    setup_logging()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")

    full_start_date = effective_full_start_date(start_date)
    end_date = clean_yyyymmdd(end_date)
    stocks = get_stock_list()
    logging.info("Start %s daily import into MySQL; stocks=%s end=%s", mode, len(stocks), end_date)

    success_count = 0
    skip_count = 0
    fail_count = 0
    total_rows = 0
    with mysql_connect() as conn:
        upsert_stock_info(conn, stocks)
        sync_stock_latest_from_dkandles(conn, ktype)
        latest_map = load_latest_update_map(conn, ktype)

        fetch_tasks = []
        for row in stocks.itertuples(index=False):
            symbol = normalize_symbol(getattr(row, "code"))
            symbol_start = full_start_date
            next_start = next_yyyymmdd(latest_map.get(symbol))
            if next_start is not None:
                symbol_start = max(next_start, full_start_date)

            if symbol_start > end_date:
                skip_count += 1
                continue

            fetch_tasks.append((symbol, symbol_start))

        logging.info("Prepared %s fetch tasks; skipped already up-to-date stocks=%s; workers=%s", len(fetch_tasks), skip_count, workers)

        def fetch_task(symbol: str, symbol_start: str) -> tuple[str, str, pd.DataFrame]:
            cfg = CrawlConfig(
                start_date=symbol_start,
                end_date=end_date,
                adjust=adjust,
                sleep_seconds=sleep_seconds,
                retries=retries,
                ktype=ktype,
            )
            df = fetch_daily_from_tencent(symbol, cfg)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return symbol, symbol_start, df

        max_workers = max(1, workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(fetch_task, symbol, symbol_start): symbol
                for symbol, symbol_start in fetch_tasks
            }
            for future in tqdm(as_completed(future_to_symbol), total=len(future_to_symbol), desc=f"{mode} daily -> mysql"):
                symbol = future_to_symbol[future]
                try:
                    symbol, symbol_start, df = future.result()
                except Exception:
                    fail_count += 1
                    logging.exception("daily %s fetch task failed", symbol)
                    continue

                if df.empty:
                    fail_count += 1
                    continue

                warmup_df = load_ma_warmup(conn, symbol, ktype, symbol_start)
                df = add_moving_averages(df, warmup_df)
                try:
                    inserted = insert_rows(conn, rows_for_insert(df))
                    latest_k_time = df["KTime"].max().to_pydatetime()
                    update_stock_latest(conn, symbol, latest_k_time)
                    conn.commit()
                    total_rows += inserted
                    success_count += 1
                except Exception:
                    conn.rollback()
                    fail_count += 1
                    logging.exception("daily %s database write failed", symbol)

    logging.info(
        "Finished %s MySQL import: success=%s skipped=%s failed=%s rows=%s",
        mode,
        success_count,
        skip_count,
        fail_count,
        total_rows,
    )


DERIVED_KLINE_CONFIG = {
    "weekly": {"table": "wkandles", "ktype": "W", "freq": "W-FRI", "time": "17:00:00"},
    "monthly": {"table": "mkandles", "ktype": "M", "freq": "ME", "time": "18:00:00"},
}


def rows_for_derived_insert(df: pd.DataFrame) -> list[tuple]:
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            (
                row.SCode,
                row.KType,
                row.KTime.to_pydatetime(),
                none_if_nan(row.Amount),
                none_if_nan(row.Volume),
                none_if_nan(row.MA5),
                none_if_nan(row.MA8),
                none_if_nan(row.MA13),
                none_if_nan(row.Open),
                none_if_nan(row.Close),
                none_if_nan(row.High),
                none_if_nan(row.Low),
                row.CreatedOn,
                row.UpdatedOn,
                none_if_nan(row.MA55),
                none_if_nan(row.MA34),
            )
        )
    return rows


def insert_derived_rows(conn: pymysql.connections.Connection, table: str, rows: list[tuple]) -> int:
    if not rows:
        return 0
    if table not in {"wkandles", "mkandles"}:
        raise ValueError(f"unsupported derived K-line table: {table}")
    insert_sql = f"""
        INSERT INTO {table}
            (SCode, KType, KTime, Amount, Volume, MA5, MA8, MA13, Open, Close, High, Low, CreatedOn, UpdatedOn, MA55, MA34)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    inserted = 0
    with conn.cursor() as cur:
        for batch in iter_batches(rows, BATCH_SIZE):
            cur.executemany(insert_sql, batch)
            inserted += len(batch)
    return inserted


def aggregate_daily_to_period(daily_df: pd.DataFrame, period: str) -> pd.DataFrame:
    cfg = DERIVED_KLINE_CONFIG[period]
    if daily_df.empty:
        return pd.DataFrame()
    source = daily_df.copy()
    source["KTime"] = pd.to_datetime(source["KTime"], errors="coerce")
    for column in ("Amount", "Volume", "Open", "Close", "High", "Low"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(subset=["KTime", "Open", "Close", "High", "Low", "Volume"])
    if source.empty:
        return pd.DataFrame()

    source = source.sort_values("KTime")
    if period == "weekly":
        source["GroupKey"] = source["KTime"].dt.to_period("W-FRI")
    else:
        source["GroupKey"] = source["KTime"].dt.to_period("M")

    grouped = source.groupby("GroupKey", sort=True)
    aggregated = grouped.agg(
        SCode=("SCode", "last"),
        Amount=("Amount", "sum"),
        Volume=("Volume", "sum"),
        Open=("Open", "first"),
        Close=("Close", "last"),
        High=("High", "max"),
        Low=("Low", "min"),
        LastDailyTime=("KTime", "max"),
    ).reset_index()
    aggregated = aggregated.dropna(subset=["SCode", "Open", "Close", "High", "Low"], how="any")
    if aggregated.empty:
        return pd.DataFrame()

    close_time = pd.to_timedelta(cfg["time"])
    if period == "weekly":
        aggregated["KTime"] = aggregated["GroupKey"].dt.end_time.dt.normalize() + close_time
    else:
        aggregated["KTime"] = pd.to_datetime(aggregated["LastDailyTime"], errors="coerce").dt.normalize() + close_time
    aggregated = aggregated.drop(columns=["GroupKey", "LastDailyTime"])
    aggregated["KType"] = cfg["ktype"]
    for window in MA_WINDOWS:
        aggregated[f"MA{window}"] = aggregated["Close"].rolling(window=window, min_periods=window).mean()
    now = datetime.now()
    aggregated["CreatedOn"] = now
    aggregated["UpdatedOn"] = now
    return aggregated

def load_daily_for_symbol(conn: pymysql.connections.Connection, symbol: str) -> pd.DataFrame:
    sql = """
        SELECT SCode, KTime, Amount, Volume, Open, Close, High, Low
        FROM dkandles
        WHERE SCode = %s AND KType = 'D'
        ORDER BY KTime
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol,))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["SCode", "KTime", "Amount", "Volume", "Open", "Close", "High", "Low"])


def generate_derived_kline(period: str) -> None:
    setup_logging()
    periods = ["weekly", "monthly"] if period == "all" else [period]
    for item in periods:
        if item not in DERIVED_KLINE_CONFIG:
            raise ValueError(f"period must be one of {sorted(GENERATE_PERIODS)}")

    with mysql_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT SCode FROM dkandles WHERE KType = 'D' ORDER BY SCode")
            symbols = [row[0] for row in cur.fetchall()]

        for item in periods:
            cfg = DERIVED_KLINE_CONFIG[item]
            table = cfg["table"]
            total_rows = 0
            success_count = 0
            fail_count = 0
            logging.info("Start generating %s K-lines into %s from dkandles; stocks=%s", item, table, len(symbols))
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {table}")
            conn.commit()

            for symbol in tqdm(symbols, total=len(symbols), desc=f"generate {item}"):
                try:
                    daily_df = load_daily_for_symbol(conn, symbol)
                    derived_df = aggregate_daily_to_period(daily_df, item)
                    inserted = insert_derived_rows(conn, table, rows_for_derived_insert(derived_df))
                    conn.commit()
                    total_rows += inserted
                    success_count += 1
                except Exception:
                    conn.rollback()
                    fail_count += 1
                    logging.exception("generate %s %s failed", item, symbol)

            logging.info("Finished generating %s: success=%s failed=%s rows=%s", item, success_count, fail_count, total_rows)


def get_trade_calendar_dates() -> list[date]:
    try:
        calendar = ak.tool_trade_date_hist_sina()
        date_col = "trade_date" if "trade_date" in calendar.columns else calendar.columns[0]
        dates = pd.to_datetime(calendar[date_col], errors="coerce").dropna().dt.date.tolist()
        return sorted(set(dates))
    except Exception:
        logging.exception("failed to fetch trade calendar")
        return []


def is_last_trade_day(today: date | None = None) -> bool:
    today = today or date.today()
    dates = [item for item in get_trade_calendar_dates() if item >= today]
    if not dates or dates[0] != today:
        return False
    return len(dates) == 1 or dates[1].month != today.month


def scheduled_weekly_generation() -> None:
    generate_derived_kline("weekly")


def scheduled_monthly_generation() -> None:
    setup_logging()
    if is_last_trade_day():
        generate_derived_kline("monthly")
    else:
        logging.info("Today is not the last trading day of the month; skip monthly generation")

def scheduled_crawl(mode: str, start_date: str, adjust: str, sleep_seconds: float, retries: int, ktype: str, workers: int) -> None:
    crawl_daily_to_mysql(
        mode=mode,
        start_date=start_date,
        end_date=date.today().strftime("%Y%m%d"),
        adjust=adjust,
        sleep_seconds=sleep_seconds,
        retries=retries,
        ktype=ktype,
        workers=workers,
    )


def run_scheduler(mode: str, start_date: str, adjust: str, sleep_seconds: float, retries: int, ktype: str, workers: int) -> None:
    setup_logging()
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    scheduler.add_job(
        scheduled_crawl,
        "cron",
        hour=15,
        minute=5,
        args=[mode, start_date, adjust, sleep_seconds, retries, ktype, workers],
        id="a_share_daily_mysql_after_close",
        replace_existing=True,
    )
    scheduler.add_job(
        scheduled_weekly_generation,
        "cron",
        day_of_week="fri",
        hour=17,
        minute=0,
        id="a_share_weekly_generation",
        replace_existing=True,
    )
    scheduler.add_job(
        scheduled_monthly_generation,
        "cron",
        hour=18,
        minute=0,
        id="a_share_monthly_generation_last_trade_day",
        replace_existing=True,
    )
    logging.info("Scheduler started: daily 15:05, weekly Friday 17:00, monthly last trading day 18:00; daily mode=%s", mode)
    scheduler.start()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch A-share daily K-line data and write directly to MySQL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one MySQL import now")
    run_parser.add_argument("--period", choices=sorted(RUN_PERIODS), default="daily", help="Only daily is supported")
    run_parser.add_argument("--mode", choices=sorted(MODES), default="incremental", help="full resumes unfinished stocks without clearing; incremental also fetches after stockinfo.LatestUpdateKandle")
    run_parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date for full mode; earliest allowed is 20100101")
    run_parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="End date, format YYYYMMDD")
    run_parser.add_argument("--adjust", default=DEFAULT_ADJUST, choices=["", "qfq", "hfq"])
    run_parser.add_argument("--sleep", type=float, default=0.05)
    run_parser.add_argument("--retries", type=int, default=3)
    run_parser.add_argument("--workers", type=int, default=8, help="Number of concurrent fetch worker threads")
    run_parser.add_argument("--ktype", default=os.environ.get("DKANDLES_KTYPE", DEFAULT_KTYPE))
    run_parser.add_argument("--use-env-proxy", action="store_true", help="Use HTTP_PROXY/HTTPS_PROXY/ALL_PROXY from the environment")

    schedule_parser = subparsers.add_parser("schedule", help="Run daily MySQL import on a schedule")
    schedule_parser.add_argument("--mode", choices=sorted(MODES), default="incremental")
    schedule_parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    schedule_parser.add_argument("--adjust", default=DEFAULT_ADJUST, choices=["", "qfq", "hfq"])
    schedule_parser.add_argument("--sleep", type=float, default=0.05)
    schedule_parser.add_argument("--retries", type=int, default=3)
    schedule_parser.add_argument("--workers", type=int, default=8, help="Number of concurrent fetch worker threads")
    schedule_parser.add_argument("--ktype", default=os.environ.get("DKANDLES_KTYPE", DEFAULT_KTYPE))
    schedule_parser.add_argument("--use-env-proxy", action="store_true")

    generate_parser = subparsers.add_parser("generate", help="Generate weekly/monthly K-lines from existing dkandles")
    generate_parser.add_argument("--period", choices=sorted(GENERATE_PERIODS), required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.use_env_proxy:
        restore_requests_proxy_handling()
    else:
        force_requests_direct()

    if args.command == "run":
        crawl_daily_to_mysql(
            mode=args.mode,
            start_date=args.start_date,
            end_date=args.end_date,
            adjust=args.adjust,
            sleep_seconds=args.sleep,
            retries=args.retries,
            ktype=args.ktype,
            workers=args.workers,
        )
    elif args.command == "generate":
        generate_derived_kline(args.period)
    elif args.command == "schedule":
        run_scheduler(
            mode=args.mode,
            start_date=args.start_date,
            adjust=args.adjust,
            sleep_seconds=args.sleep,
            retries=args.retries,
            ktype=args.ktype,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
