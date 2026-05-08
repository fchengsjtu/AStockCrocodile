from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import pymysql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "daily"
BATCH_SIZE = 5000
MA_WINDOWS = (5, 8, 13, 34, 55)


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=env("MYSQL_HOST"),
        port=int(env("MYSQL_PORT", "3306")),
        user=env("MYSQL_USER"),
        password=env("MYSQL_PASSWORD"),
        database=env("MYSQL_DATABASE"),
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=15,
        read_timeout=60,
        write_timeout=60,
    )


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def normalized_symbol(csv_path: Path, df: pd.DataFrame) -> str:
    col = pick_column(df, ("股票代码", "代码", "SCode"))
    if col:
        series = df[col].dropna().astype(str)
        if not series.empty:
            digits = "".join(ch for ch in series.iloc[0] if ch.isdigit())
            if digits:
                return digits[-6:].zfill(6)
    return csv_path.stem.zfill(6)


def load_csv(csv_path: Path, ktype: str) -> list[tuple]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return []

    column_map = {
        "date": pick_column(df, ("日期", "KTime", "date")),
        "open": pick_column(df, ("开盘", "Open", "open")),
        "close": pick_column(df, ("收盘", "Close", "close")),
        "high": pick_column(df, ("最高", "High", "high")),
        "low": pick_column(df, ("最低", "Low", "low")),
        "volume": pick_column(df, ("成交量", "Volume", "volume", "amount")),
        "amount": pick_column(df, ("成交额", "Amount", "turnover")),
    }
    missing = [key for key in ("date", "open", "close", "high", "low", "volume") if column_map[key] is None]
    if missing:
        raise ValueError(f"{csv_path.name} missing required columns: {missing}")

    symbol = normalized_symbol(csv_path, df)
    out = pd.DataFrame(
        {
            "SCode": symbol,
            "KType": ktype,
            "KTime": pd.to_datetime(df[column_map["date"]], errors="coerce"),
            "Open": pd.to_numeric(df[column_map["open"]], errors="coerce"),
            "Close": pd.to_numeric(df[column_map["close"]], errors="coerce"),
            "High": pd.to_numeric(df[column_map["high"]], errors="coerce"),
            "Low": pd.to_numeric(df[column_map["low"]], errors="coerce"),
            "Volume": pd.to_numeric(df[column_map["volume"]], errors="coerce"),
            "Amount": 0 if column_map["amount"] is None else pd.to_numeric(df[column_map["amount"]], errors="coerce"),
        }
    )
    out = out.dropna(subset=["KTime", "Open", "Close", "High", "Low", "Volume"])
    out = out.sort_values("KTime").drop_duplicates(subset=["SCode", "KType", "KTime"], keep="last")
    for window in MA_WINDOWS:
        out[f"MA{window}"] = out["Close"].rolling(window=window, min_periods=window).mean()
    out["CreatedOn"] = datetime.now()
    out["UpdatedOn"] = out["CreatedOn"]

    rows = []
    for row in out.itertuples(index=False):
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


def none_if_nan(value):
    if pd.isna(value):
        return None
    return value


def iter_batches(items: list[tuple], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def import_daily(data_dir: Path, clear: bool, ktype: str) -> None:
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {data_dir}")

    insert_sql = """
        INSERT INTO mkandles
            (SCode, KType, KTime, Amount, Volume, MA5, MA8, MA13, Open, Close, High, Low, CreatedOn, UpdatedOn, MA55, MA34)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    total_rows = 0
    with connect() as conn:
        with conn.cursor() as cur:
            if clear:
                cur.execute("DELETE FROM mkandles")
                conn.commit()
                print("Cleared mkandles")

            for index, csv_path in enumerate(csv_files, start=1):
                rows = load_csv(csv_path, ktype=ktype)
                for batch in iter_batches(rows, BATCH_SIZE):
                    cur.executemany(insert_sql, batch)
                    total_rows += len(batch)
                conn.commit()
                if index % 25 == 0 or index == len(csv_files):
                    print(f"Imported {index}/{len(csv_files)} files, {total_rows} rows")

        conn.commit()
    print(f"Done. Imported {total_rows} rows from {len(csv_files)} files into mkandles")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import local daily K-line CSV files into MySQL mkandles")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-clear", action="store_true", help="Do not delete existing rows before import")
    parser.add_argument("--ktype", default=os.environ.get("MKANDLES_KTYPE", "M"))
    args = parser.parse_args()
    import_daily(args.data_dir, clear=not args.no_clear, ktype=args.ktype)


if __name__ == "__main__":
    main()