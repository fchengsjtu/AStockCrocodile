from __future__ import annotations

import argparse
import hashlib
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import combinations
from typing import Iterable

import pandas as pd
import pymysql

from a_share_crawler import DEFAULT_KTYPE, mysql_connect, none_if_nan
from kline_statistics import SHORT_TERM_SURGE_TYPE
from stock_selector import parse_date

SURGE_PATTERN_TABLE = "surgepatterns"
DEFAULT_BATCH_SIZE = 40
DEFAULT_DAILY_WINDOW = 56
DEFAULT_WEEKLY_WINDOW = 56

PATTERN_COLUMNS = [
    "Pattern",
    "FeatureCount",
    "SampleCount",
    "SuccessCount",
    "SuccessRate",
    "PositiveSupport",
]


@dataclass(frozen=True)
class SurgePatternConfig:
    start_date: str
    end_date: str
    stat_type: str
    min_success_rate: float
    min_sample_count: int
    min_positive_support: int
    max_pattern_size: int
    daily_window: int
    weekly_window: int
    batch_size: int
    output: str | None
    save_db: bool


def iter_batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def ensure_surge_pattern_table(conn: pymysql.connections.Connection) -> None:
    sql = f"""
        CREATE TABLE IF NOT EXISTS {SURGE_PATTERN_TABLE} (
            Id BIGINT NOT NULL AUTO_INCREMENT,
            PatternHash CHAR(64) NOT NULL,
            PatternText TEXT NOT NULL,
            StatType VARCHAR(64) NOT NULL,
            StartDate DATE NOT NULL,
            EndDate DATE NOT NULL,
            FeatureCount INT NOT NULL,
            SampleCount INT NOT NULL,
            SuccessCount INT NOT NULL,
            SuccessRate DECIMAL(18,6) NOT NULL,
            PositiveSupport INT NOT NULL,
            CreatedOn DATETIME NOT NULL,
            PRIMARY KEY (Id),
            UNIQUE KEY ux_surgepatterns_range_hash (StatType, StartDate, EndDate, PatternHash),
            KEY idx_surgepatterns_rate (StatType, SuccessRate)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def load_positive_events(conn: pymysql.connections.Connection, stat_type: str, start_date: date, end_date: date) -> pd.DataFrame:
    sql = """
        SELECT SCode, SName, StartRiseDate, PrevTradeDate,
               COALESCE(SelectionDate, PrevTradeDate) AS SelectionDate,
               GainRate
        FROM klinestatistics
        WHERE StatType = %s
          AND COALESCE(SelectionDate, PrevTradeDate) >= %s
          AND COALESCE(SelectionDate, PrevTradeDate) <= %s
        ORDER BY SCode, SelectionDate
    """
    with conn.cursor() as cur:
        cur.execute(sql, (stat_type, start_date, end_date))
        rows = cur.fetchall()
    columns = ["SCode", "SName", "StartRiseDate", "PrevTradeDate", "SelectionDate", "GainRate"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    for column in ["StartRiseDate", "PrevTradeDate", "SelectionDate"]:
        df[column] = pd.to_datetime(df[column]).dt.date
    return df


def load_symbols(conn: pymysql.connections.Connection, start_date: date, end_date: date, ktype: str = DEFAULT_KTYPE) -> list[str]:
    sql = """
        SELECT DISTINCT SCode
        FROM dkandles
        WHERE KType = %s AND KTime >= %s AND KTime < %s
        ORDER BY SCode
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ktype, start_date, end_date + timedelta(days=1)))
        rows = cur.fetchall()
    return [row[0] for row in rows]


def load_kline_for_symbols(
    conn: pymysql.connections.Connection,
    table: str,
    ktype: str,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Volume", "Amount", "MA5", "MA13", "MA34", "MA55"])
    placeholders = ",".join(["%s"] * len(symbols))
    sql = f"""
        SELECT k.SCode, si.SName, DATE(k.KTime) AS TradeDate,
               k.Open, k.Close, k.High, k.Low, k.Volume, k.Amount, k.MA5, k.MA13, k.MA34, k.MA55
        FROM {table} k
        LEFT JOIN stockinfo si ON si.SCode = k.SCode
        WHERE k.KType = %s
          AND k.SCode IN ({placeholders})
          AND k.KTime >= %s AND k.KTime < %s
        ORDER BY k.SCode, k.KTime
    """
    params = [ktype, *symbols, start_date, end_date + timedelta(days=1)]
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    columns = ["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Volume", "Amount", "MA5", "MA13", "MA34", "MA55"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    numeric_columns = ["Open", "Close", "High", "Low", "Volume", "Amount", "MA5", "MA13", "MA34", "MA55"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _safe_ratio(new_value: float | None, old_value: float | None) -> float | None:
    if new_value is None or old_value is None or pd.isna(new_value) or pd.isna(old_value) or old_value <= 0:
        return None
    return float(new_value / old_value - 1)


def _safe_mean(series: pd.Series) -> float | None:
    value = series.mean(skipna=True)
    if pd.isna(value):
        return None
    return float(value)


def _add_return_features(features: set[str], prefix: str, close: float, window: pd.DataFrame, periods: list[int]) -> None:
    for period in periods:
        if len(window) <= period:
            continue
        rate = _safe_ratio(close, window.iloc[-period - 1]["Close"])
        if rate is None:
            continue
        if rate >= 0.20:
            features.add(f"{prefix}_RET_{period}_GE_20")
        elif rate >= 0.10:
            features.add(f"{prefix}_RET_{period}_GE_10")
        elif rate >= 0.05:
            features.add(f"{prefix}_RET_{period}_GE_5")
        elif rate <= -0.15:
            features.add(f"{prefix}_RET_{period}_LE_-15")
        elif rate <= -0.08:
            features.add(f"{prefix}_RET_{period}_LE_-8")
        else:
            features.add(f"{prefix}_RET_{period}_RANGE_-8_5")


def _add_ma_features(features: set[str], prefix: str, item: pd.Series) -> None:
    close = item["Close"]
    for ma in ["MA5", "MA13", "MA34", "MA55"]:
        value = item.get(ma)
        if pd.notna(close) and pd.notna(value):
            features.add(f"{prefix}_CLOSE_GT_{ma}" if close > value else f"{prefix}_CLOSE_LE_{ma}")
    if pd.notna(item.get("MA5")) and pd.notna(item.get("MA13")):
        features.add(f"{prefix}_MA5_GT_MA13" if item["MA5"] > item["MA13"] else f"{prefix}_MA5_LE_MA13")
    if pd.notna(item.get("MA13")) and pd.notna(item.get("MA34")):
        features.add(f"{prefix}_MA13_GT_MA34" if item["MA13"] > item["MA34"] else f"{prefix}_MA13_LE_MA34")
    if pd.notna(item.get("MA34")) and pd.notna(item.get("MA55")):
        features.add(f"{prefix}_MA34_GT_MA55" if item["MA34"] > item["MA55"] else f"{prefix}_MA34_LE_MA55")


def _add_volume_features(features: set[str], prefix: str, window: pd.DataFrame, recent_size: int, base_size: int) -> None:
    if len(window) < recent_size + base_size:
        return
    recent = _safe_mean(window.iloc[-recent_size:]["Volume"])
    base = _safe_mean(window.iloc[-recent_size - base_size : -recent_size]["Volume"])
    if recent is None or base is None or base <= 0:
        return
    ratio = recent / base
    if ratio >= 2.0:
        features.add(f"{prefix}_VOL_RATIO_GE_2")
    elif ratio >= 1.5:
        features.add(f"{prefix}_VOL_RATIO_GE_1_5")
    elif ratio <= 0.7:
        features.add(f"{prefix}_VOL_RATIO_LE_0_7")
    else:
        features.add(f"{prefix}_VOL_RATIO_NORMAL")


def build_pattern_features(daily_window: pd.DataFrame, weekly_window: pd.DataFrame) -> set[str]:
    if len(daily_window) < 2 or len(weekly_window) < 2:
        return set()
    daily = daily_window.sort_values("TradeDate").reset_index(drop=True)
    weekly = weekly_window.sort_values("TradeDate").reset_index(drop=True)
    features: set[str] = set()

    d_last = daily.iloc[-1]
    w_last = weekly.iloc[-1]
    d_close = float(d_last["Close"])
    w_close = float(w_last["Close"])

    _add_ma_features(features, "D", d_last)
    _add_ma_features(features, "W", w_last)
    _add_return_features(features, "D", d_close, daily, [5, 10, 20, 55])
    _add_return_features(features, "W", w_close, weekly, [4, 8, 13, 26, 55])
    _add_volume_features(features, "D", daily, recent_size=5, base_size=20)
    _add_volume_features(features, "W", weekly, recent_size=4, base_size=12)

    d_high_55 = daily["High"].max(skipna=True)
    d_low_55 = daily["Low"].min(skipna=True)
    if pd.notna(d_high_55) and d_high_55 > 0:
        features.add("D_CLOSE_NEAR_55D_HIGH" if d_close >= d_high_55 * 0.95 else "D_CLOSE_NOT_NEAR_55D_HIGH")
    if pd.notna(d_high_55) and pd.notna(d_low_55):
        midpoint = (float(d_high_55) + float(d_low_55)) / 2
        features.add("D_CLOSE_ABOVE_55D_MID" if d_close >= midpoint else "D_CLOSE_BELOW_55D_MID")

    w_high_55 = weekly["High"].max(skipna=True)
    w_low_55 = weekly["Low"].min(skipna=True)
    if pd.notna(w_high_55) and w_high_55 > 0:
        features.add("W_CLOSE_NEAR_55W_HIGH" if w_close >= w_high_55 * 0.95 else "W_CLOSE_NOT_NEAR_55W_HIGH")
    if pd.notna(w_high_55) and pd.notna(w_low_55):
        midpoint = (float(w_high_55) + float(w_low_55)) / 2
        features.add("W_CLOSE_ABOVE_55W_MID" if w_close >= midpoint else "W_CLOSE_BELOW_55W_MID")

    if len(daily) >= 30:
        recent_range = _safe_mean((daily.iloc[-10:]["High"] - daily.iloc[-10:]["Low"]) / daily.iloc[-10:]["Close"])
        prior_range = _safe_mean((daily.iloc[-30:-10]["High"] - daily.iloc[-30:-10]["Low"]) / daily.iloc[-30:-10]["Close"])
        if recent_range is not None and prior_range is not None and prior_range > 0:
            features.add("D_RANGE_10_CONTRACT" if recent_range <= prior_range * 0.8 else "D_RANGE_10_NOT_CONTRACT")

    return features


def iter_pattern_keys(features: set[str], max_pattern_size: int) -> Iterable[tuple[str, ...]]:
    ordered = sorted(features)
    for size in range(1, max_pattern_size + 1):
        yield from combinations(ordered, size)


def pattern_to_text(pattern: tuple[str, ...]) -> str:
    return " && ".join(pattern)


def pattern_hash(pattern: tuple[str, ...]) -> str:
    return hashlib.sha256(pattern_to_text(pattern).encode("utf-8")).hexdigest()


def make_frame_map(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {}
    if df.empty:
        return frames
    for symbol, group in df.groupby("SCode", sort=False):
        frames[symbol] = group.sort_values("TradeDate").reset_index(drop=True)
    return frames


def extract_features_for_date(
    daily_frame: pd.DataFrame,
    weekly_frame: pd.DataFrame,
    trade_date: date,
    daily_window_size: int,
    weekly_window_size: int,
) -> set[str] | None:
    daily_matches = daily_frame.index[daily_frame["TradeDate"] == trade_date].tolist()
    if not daily_matches:
        return None
    return extract_features_at_position(
        daily_frame,
        weekly_frame,
        daily_matches[0],
        trade_date,
        daily_window_size,
        weekly_window_size,
    )


def extract_features_at_position(
    daily_frame: pd.DataFrame,
    weekly_frame: pd.DataFrame,
    daily_pos: int,
    trade_date: date,
    daily_window_size: int,
    weekly_window_size: int,
) -> set[str] | None:
    if daily_pos + 1 < daily_window_size:
        return None
    weekly_dates = weekly_frame["TradeDate"].tolist()
    weekly_pos = bisect_right(weekly_dates, trade_date) - 1
    if weekly_pos + 1 < weekly_window_size:
        return None
    daily_window = daily_frame.iloc[daily_pos + 1 - daily_window_size : daily_pos + 1]
    weekly_window = weekly_frame.iloc[weekly_pos + 1 - weekly_window_size : weekly_pos + 1]
    return build_pattern_features(daily_window, weekly_window)


def mine_positive_patterns(
    conn: pymysql.connections.Connection,
    positives: pd.DataFrame,
    config: SurgePatternConfig,
    start_date: date,
    end_date: date,
) -> dict[tuple[str, ...], int]:
    support: dict[tuple[str, ...], int] = defaultdict(int)
    if positives.empty:
        return support
    symbols = sorted(positives["SCode"].dropna().unique().tolist())
    lookback_start = start_date - timedelta(days=max(500, config.weekly_window * 10, config.daily_window * 3))
    for batch_index, batch in enumerate(iter_batches(symbols, config.batch_size), start=1):
        daily_df = load_kline_for_symbols(conn, "dkandles", DEFAULT_KTYPE, batch, lookback_start, end_date)
        weekly_df = load_kline_for_symbols(conn, "wkandles", "W", batch, lookback_start, end_date)
        daily_frames = make_frame_map(daily_df)
        weekly_frames = make_frame_map(weekly_df)
        batch_events = positives[positives["SCode"].isin(batch)]
        feature_rows = 0
        for event in batch_events.itertuples(index=False):
            daily_frame = daily_frames.get(event.SCode)
            weekly_frame = weekly_frames.get(event.SCode)
            if daily_frame is None or weekly_frame is None:
                continue
            features = extract_features_for_date(daily_frame, weekly_frame, event.SelectionDate, config.daily_window, config.weekly_window)
            if not features:
                continue
            feature_rows += 1
            for pattern in iter_pattern_keys(features, config.max_pattern_size):
                support[pattern] += 1
        print(
            f"positive batch {batch_index} symbols={len(batch)} events={len(batch_events)} "
            f"feature_rows={feature_rows} candidate_patterns={len(support)}",
            flush=True,
        )
    return support


def evaluate_patterns(
    conn: pymysql.connections.Connection,
    positives: pd.DataFrame,
    target_patterns: set[tuple[str, ...]],
    positive_support: dict[tuple[str, ...], int],
    config: SurgePatternConfig,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if not target_patterns:
        return pd.DataFrame(columns=PATTERN_COLUMNS)
    positive_keys = set(zip(positives["SCode"], positives["SelectionDate"]))
    counts: dict[tuple[str, ...], list[int]] = {pattern: [0, 0] for pattern in target_patterns}
    lookback_start = start_date - timedelta(days=max(500, config.weekly_window * 10, config.daily_window * 3))
    symbols = load_symbols(conn, start_date, end_date)
    batches = list(iter_batches(symbols, config.batch_size))
    print(
        f"evaluate patterns symbols={len(symbols)} batches={len(batches)} "
        f"target_patterns={len(target_patterns)}",
        flush=True,
    )
    for batch_index, batch in enumerate(batches, start=1):
        daily_df = load_kline_for_symbols(conn, "dkandles", DEFAULT_KTYPE, batch, lookback_start, end_date)
        weekly_df = load_kline_for_symbols(conn, "wkandles", "W", batch, lookback_start, end_date)
        daily_frames = make_frame_map(daily_df)
        weekly_frames = make_frame_map(weekly_df)
        scanned = 0
        matched = 0
        for symbol in batch:
            daily_frame = daily_frames.get(symbol)
            weekly_frame = weekly_frames.get(symbol)
            if daily_frame is None or weekly_frame is None:
                continue
            date_rows = daily_frame[(daily_frame["TradeDate"] >= start_date) & (daily_frame["TradeDate"] <= end_date)]
            for daily_pos, item in date_rows.iterrows():
                features = extract_features_at_position(
                    daily_frame,
                    weekly_frame,
                    int(daily_pos),
                    item.TradeDate,
                    config.daily_window,
                    config.weekly_window,
                )
                if not features:
                    continue
                scanned += 1
                is_success = (symbol, item.TradeDate) in positive_keys
                for pattern in iter_pattern_keys(features, config.max_pattern_size):
                    if pattern not in target_patterns:
                        continue
                    counts[pattern][0] += 1
                    if is_success:
                        counts[pattern][1] += 1
                    matched += 1
        print(
            f"evaluate batch {batch_index}/{len(batches)} symbols={len(batch)} "
            f"scanned_dates={scanned} pattern_hits={matched}",
            flush=True,
        )

    rows = []
    for pattern, (sample_count, success_count) in counts.items():
        if sample_count < config.min_sample_count:
            continue
        success_rate = success_count / sample_count if sample_count else 0.0
        if success_rate < config.min_success_rate:
            continue
        rows.append(
            {
                "Pattern": pattern_to_text(pattern),
                "FeatureCount": len(pattern),
                "SampleCount": sample_count,
                "SuccessCount": success_count,
                "SuccessRate": success_rate,
                "PositiveSupport": positive_support.get(pattern, 0),
            }
        )
    if not rows:
        return pd.DataFrame(columns=PATTERN_COLUMNS)
    return pd.DataFrame(rows, columns=PATTERN_COLUMNS).sort_values(
        ["SuccessRate", "SuccessCount", "SampleCount"], ascending=[False, False, False]
    )


def save_patterns(conn: pymysql.connections.Connection, patterns: pd.DataFrame, config: SurgePatternConfig, start_date: date, end_date: date) -> int:
    ensure_surge_pattern_table(conn)
    if patterns.empty:
        return 0
    now = datetime.now()
    rows = []
    for row in patterns.itertuples(index=False):
        pattern_tuple = tuple(str(row.Pattern).split(" && "))
        rows.append(
            (
                pattern_hash(pattern_tuple),
                row.Pattern,
                config.stat_type,
                start_date,
                end_date,
                int(row.FeatureCount),
                int(row.SampleCount),
                int(row.SuccessCount),
                none_if_nan(row.SuccessRate),
                int(row.PositiveSupport),
                now,
            )
        )
    sql = f"""
        INSERT INTO {SURGE_PATTERN_TABLE}
            (PatternHash, PatternText, StatType, StartDate, EndDate, FeatureCount,
             SampleCount, SuccessCount, SuccessRate, PositiveSupport, CreatedOn)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            PatternText = VALUES(PatternText),
            FeatureCount = VALUES(FeatureCount),
            SampleCount = VALUES(SampleCount),
            SuccessCount = VALUES(SuccessCount),
            SuccessRate = VALUES(SuccessRate),
            PositiveSupport = VALUES(PositiveSupport),
            CreatedOn = VALUES(CreatedOn)
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def run_pattern_mining(config: SurgePatternConfig) -> pd.DataFrame:
    start_date = parse_date(config.start_date)
    end_date = parse_date(config.end_date)
    if start_date is None or end_date is None:
        raise ValueError("start-date and end-date are required")
    if start_date > end_date:
        raise ValueError("start-date must be <= end-date")
    with mysql_connect() as conn:
        if config.save_db:
            ensure_surge_pattern_table(conn)
        positives = load_positive_events(conn, config.stat_type, start_date, end_date)
        print(f"loaded positives stat_type={config.stat_type} rows={len(positives)}", flush=True)
        positive_support = mine_positive_patterns(conn, positives, config, start_date, end_date)
        target_patterns = {
            pattern for pattern, support in positive_support.items() if support >= config.min_positive_support
        }
        print(
            f"positive patterns={len(positive_support)} "
            f"target_patterns={len(target_patterns)} min_positive_support={config.min_positive_support}",
            flush=True,
        )
        patterns = evaluate_patterns(conn, positives, target_patterns, positive_support, config, start_date, end_date)
        if config.output:
            patterns.to_csv(config.output, index=False, encoding="utf-8-sig")
        saved = save_patterns(conn, patterns, config, start_date, end_date) if config.save_db else 0
    print(f"surge patterns kept={len(patterns)} saved={saved} min_success_rate={config.min_success_rate:.2%}", flush=True)
    return patterns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine reusable patterns before short-term surge events")
    parser.add_argument("--start-date", required=True, help="Selection date start, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Selection date end, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--stat-type", default=SHORT_TERM_SURGE_TYPE, help="K-line statistic type from klinestatistics")
    parser.add_argument("--min-success-rate", type=float, default=0.50, help="Minimum pattern success rate")
    parser.add_argument("--min-sample-count", type=int, default=20, help="Minimum total occurrences for a pattern")
    parser.add_argument("--min-positive-support", type=int, default=5, help="Minimum positive occurrences before evaluation")
    parser.add_argument("--max-pattern-size", type=int, default=2, help="Maximum number of feature clauses in one pattern")
    parser.add_argument("--daily-window", type=int, default=DEFAULT_DAILY_WINDOW, help="Daily bars ending at SelectionDate; default is SelectionDate plus 55 prior trading days")
    parser.add_argument("--weekly-window", type=int, default=DEFAULT_WEEKLY_WINDOW, help="Weekly bars ending at SelectionDate; default is SelectionDate plus 55 prior weekly bars")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of symbols loaded per batch")
    parser.add_argument("--output", help="Optional CSV path for retained patterns")
    parser.add_argument("--no-save-db", action="store_true", help="Do not save patterns to MySQL")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_pattern_mining(
        SurgePatternConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            stat_type=args.stat_type,
            min_success_rate=args.min_success_rate,
            min_sample_count=max(1, args.min_sample_count),
            min_positive_support=max(1, args.min_positive_support),
            max_pattern_size=max(1, args.max_pattern_size),
            daily_window=max(2, args.daily_window),
            weekly_window=max(2, args.weekly_window),
            batch_size=max(1, args.batch_size),
            output=args.output,
            save_db=not args.no_save_db,
        )
    )


if __name__ == "__main__":
    main()
