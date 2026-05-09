from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import pymysql

from a_share_crawler import DEFAULT_KTYPE, mysql_connect, none_if_nan
from news_crawler import CONCEPT_KEYWORDS, MAX_RELATED_CONCEPTS, ensure_news_table

SELECTION_TABLE = "stockselection"
DEFAULT_LOOKBACK_DAYS = 140
STRATEGY_MA_BULLISH = "ma_bullish_v1"
STRATEGY_NEWS_HOT = "news_hot_v1"
STRATEGY_WEEKLY_VOLUME_DROP = "weekly_volume_drop_v1"
STRATEGIES = (STRATEGY_MA_BULLISH, STRATEGY_NEWS_HOT, STRATEGY_WEEKLY_VOLUME_DROP)


@dataclass(frozen=True)
class SelectionConfig:
    trade_date: str | None
    min_turnover_amount: float
    limit: int | None
    ktype: str
    strategy: str
    min_recommendations: int
    max_recommendations: int


def parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid date: {value}")
    return parsed.date()


def ensure_selection_table(conn: pymysql.connections.Connection) -> None:
    sql = f"""
        CREATE TABLE IF NOT EXISTS {SELECTION_TABLE} (
            Id BIGINT NOT NULL AUTO_INCREMENT,
            TradeDate DATE NOT NULL,
            SCode VARCHAR(10) NOT NULL,
            SName VARCHAR(64) NULL,
            StrategyName VARCHAR(64) NOT NULL DEFAULT 'ma_bullish_v1',
            ClosePrice DECIMAL(18,6) NOT NULL,
            Score DECIMAL(18,6) NULL,
            Reason VARCHAR(255) NULL,
            CreatedOn DATETIME NOT NULL,
            PRIMARY KEY (Id),
            UNIQUE KEY ux_stockselection_strategy_date_code (StrategyName, TradeDate, SCode),
            KEY idx_stockselection_code (SCode)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute(f"SHOW COLUMNS FROM {SELECTION_TABLE} LIKE 'StrategyName'")
        if cur.fetchone() is None:
            cur.execute(f"ALTER TABLE {SELECTION_TABLE} ADD COLUMN StrategyName VARCHAR(64) NOT NULL DEFAULT 'ma_bullish_v1' AFTER SName")
        cur.execute(f"SHOW INDEX FROM {SELECTION_TABLE} WHERE Key_name = 'ux_stockselection_date_code'")
        if cur.fetchone() is not None:
            cur.execute(f"ALTER TABLE {SELECTION_TABLE} DROP INDEX ux_stockselection_date_code")
        cur.execute(f"SHOW INDEX FROM {SELECTION_TABLE} WHERE Key_name = 'ux_stockselection_strategy_date_code'")
        if cur.fetchone() is None:
            cur.execute(f"ALTER TABLE {SELECTION_TABLE} ADD UNIQUE KEY ux_stockselection_strategy_date_code (StrategyName, TradeDate, SCode)")
    conn.commit()


def latest_trade_date(conn: pymysql.connections.Connection, ktype: str = DEFAULT_KTYPE) -> date:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(DATE(KTime)) FROM dkandles WHERE KType = %s", (ktype,))
        value = cur.fetchone()[0]
    if value is None:
        raise RuntimeError("dkandles has no daily K-line data")
    return value if isinstance(value, date) else pd.to_datetime(value).date()


def load_daily_window(conn: pymysql.connections.Connection, trade_date: date, lookback_days: int, ktype: str) -> pd.DataFrame:
    start_date = trade_date - timedelta(days=lookback_days * 2)
    sql = """
        SELECT dk.SCode, si.SName, DATE(dk.KTime) AS TradeDate,
               dk.Open, dk.Close, dk.High, dk.Low, dk.Amount, dk.Volume,
               dk.MA5, dk.MA8, dk.MA13, dk.MA34, dk.MA55
        FROM dkandles dk
        LEFT JOIN stockinfo si ON si.SCode = dk.SCode
        WHERE dk.KType = %s AND dk.KTime >= %s AND dk.KTime < %s
        ORDER BY dk.SCode, dk.KTime
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ktype, start_date, trade_date + timedelta(days=1)))
        rows = cur.fetchall()
    columns = ["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume", "MA5", "MA8", "MA13", "MA34", "MA55"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    for column in columns[3:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def compute_strategy_frame(daily_df: pd.DataFrame, min_turnover_amount: float = 0.0) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    frames = []
    for _, group in daily_df.groupby("SCode", sort=False):
        item = group.sort_values("TradeDate").copy()
        item["PrevClose"] = item["Close"].shift(1)
        item["AvgAmount5"] = item["Amount"].rolling(5, min_periods=5).mean()
        item["PctChange"] = item["Close"] / item["PrevClose"] - 1
        frames.append(item)
    df = pd.concat(frames, ignore_index=True)

    ma_ready = df[["MA5", "MA8", "MA13", "MA34", "MA55"]].notna().all(axis=1)
    ma_bullish = (df["MA5"] > df["MA8"]) & (df["MA8"] > df["MA13"]) & (df["MA13"] > df["MA34"]) & (df["MA34"] > df["MA55"])
    price_confirm = (df["Close"] > df["MA5"]) & (df["Close"] > df["Open"])
    liquidity = df["AvgAmount5"].fillna(0) >= min_turnover_amount
    not_extreme = df["PctChange"].fillna(0) < 0.095
    df["Selected"] = ma_ready & ma_bullish & price_confirm & liquidity & not_extreme
    df["Score"] = (df["Close"] / df["MA55"] - 1).where(df["MA55"] > 0)
    df["Reason"] = "MA5>MA8>MA13>MA34>MA55; close>MA5; bullish candle"
    return df


def load_weekly_window(conn: pymysql.connections.Connection, trade_date: date, lookback_weeks: int = 20) -> pd.DataFrame:
    start_date = trade_date - timedelta(days=lookback_weeks * 7 + 14)
    sql = """
        SELECT wk.SCode, si.SName, DATE(wk.KTime) AS TradeDate,
               wk.Open, wk.Close, wk.High, wk.Low, wk.Amount, wk.Volume
        FROM wkandles wk
        LEFT JOIN stockinfo si ON si.SCode = wk.SCode
        WHERE wk.KType = 'W' AND wk.KTime >= %s AND wk.KTime < %s
        ORDER BY wk.SCode, wk.KTime
    """
    with conn.cursor() as cur:
        cur.execute(sql, (start_date, trade_date + timedelta(days=1)))
        rows = cur.fetchall()
    columns = ["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.date
    for column in columns[3:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def parse_json_value(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def extract_concept_heat(value) -> dict[str, float]:
    parsed = parse_json_value(value)
    heat: dict[str, float] = {}
    for item in parsed:
        if isinstance(item, dict):
            concept = str(item.get("concept") or "").strip()
            try:
                score = float(item.get("heat") or 0)
            except (TypeError, ValueError):
                score = 0.0
        else:
            concept = str(item).strip()
            score = 0.0
        if concept:
            heat[concept] = max(heat.get(concept, 0.0), score)
    return heat


def load_news_for_date(conn: pymysql.connections.Connection, trade_date: date) -> pd.DataFrame:
    ensure_news_table(conn)
    sql = """
        SELECT DATE(PublishTime) AS NewsDate, Title, Summary, Heat, CredibilityLevel, RelatedConcepts, ConceptHeat
        FROM news
        WHERE PublishTime >= %s AND PublishTime < %s
        ORDER BY PublishTime DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (trade_date, trade_date + timedelta(days=1)))
        rows = cur.fetchall()
    if not rows:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(DATE(PublishTime)) FROM news WHERE PublishTime < %s", (trade_date + timedelta(days=1),))
            latest = cur.fetchone()[0]
        if latest is None:
            return pd.DataFrame(columns=["NewsDate", "Title", "Summary", "Heat", "CredibilityLevel", "RelatedConcepts", "ConceptHeat"])
        with conn.cursor() as cur:
            cur.execute(sql, (latest, latest + timedelta(days=1)))
            rows = cur.fetchall()
    columns = ["NewsDate", "Title", "Summary", "Heat", "CredibilityLevel", "RelatedConcepts", "ConceptHeat"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["Heat"] = pd.to_numeric(df["Heat"], errors="coerce").fillna(0)
    df["CredibilityLevel"] = pd.to_numeric(df["CredibilityLevel"], errors="coerce").fillna(5)
    return df


def concept_heat_from_news(news_df: pd.DataFrame) -> dict[str, float]:
    heat: dict[str, float] = {}
    if news_df.empty:
        return heat
    for row in news_df.itertuples(index=False):
        for concept, value in extract_concept_heat(getattr(row, "ConceptHeat", None)).items():
            heat[concept] = max(heat.get(concept, 0.0), value)
    if heat:
        return heat
    total = len(news_df)
    counts: dict[str, int] = {}
    for value in news_df["RelatedConcepts"]:
        concepts = parse_json_value(value)
        for concept in set(str(item) for item in concepts[:MAX_RELATED_CONCEPTS] if item):
            counts[concept] = counts.get(concept, 0) + 1
    return {concept: count / total for concept, count in counts.items()} if total else {}


def stock_concept_match_score(stock_name: str | None, title: str | None, summary: str | None, concept: str) -> float:
    text = f"{title or ''} {summary or ''}"
    score = 0.0
    if stock_name and stock_name in text:
        score += 1.0
    keywords = CONCEPT_KEYWORDS.get(concept, ())
    for keyword in keywords:
        if keyword and keyword in text:
            score += 0.15
    return min(score, 1.0)


def build_stock_news_scores(news_df: pd.DataFrame, stocks_df: pd.DataFrame, concept_heat: dict[str, float]) -> pd.DataFrame:
    rows = []
    if news_df.empty or stocks_df.empty:
        return pd.DataFrame(columns=["SCode", "NewsScore", "HotConcepts"])
    for stock in stocks_df.itertuples(index=False):
        concept_scores: dict[str, float] = {}
        for news in news_df.itertuples(index=False):
            concepts = parse_json_value(news.RelatedConcepts)
            credibility_weight = (11 - float(news.CredibilityLevel)) / 10
            heat_weight = 1 + min(float(news.Heat or 0), 100000) / 100000
            for rank, concept in enumerate(concepts[:MAX_RELATED_CONCEPTS], start=1):
                concept = str(concept)
                relation_weight = (MAX_RELATED_CONCEPTS - rank + 1) / MAX_RELATED_CONCEPTS
                text_match = stock_concept_match_score(stock.SName, news.Title, news.Summary, concept)
                if text_match <= 0:
                    continue
                score = concept_heat.get(concept, 0.0) * relation_weight * credibility_weight * heat_weight * text_match
                concept_scores[concept] = concept_scores.get(concept, 0.0) + score
        if concept_scores:
            ordered = sorted(concept_scores.items(), key=lambda item: (-item[1], item[0]))
            rows.append(
                {
                    "SCode": stock.SCode,
                    "NewsScore": sum(concept_scores.values()),
                    "HotConcepts": ",".join(concept for concept, _ in ordered[:3]),
                }
            )
    return pd.DataFrame(rows, columns=["SCode", "NewsScore", "HotConcepts"])


def compute_performance_score(daily_df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    rows = []
    if daily_df.empty:
        return pd.DataFrame(columns=["SCode", "PerformanceScore"])
    for symbol, group in daily_df.groupby("SCode", sort=False):
        frame = group.sort_values("TradeDate").reset_index(drop=True)
        current = frame[frame["TradeDate"] == trade_date]
        if current.empty:
            continue
        pos = int(current.index[0])
        row = frame.iloc[pos]
        pct5 = 0.0
        pct20 = 0.0
        if pos >= 5 and frame.iloc[pos - 5]["Close"] > 0:
            pct5 = float(row["Close"] / frame.iloc[pos - 5]["Close"] - 1)
        if pos >= 20 and frame.iloc[pos - 20]["Close"] > 0:
            pct20 = float(row["Close"] / frame.iloc[pos - 20]["Close"] - 1)
        volume_score = 0.0
        if pd.notna(row.get("AvgAmount5", None)) and row["AvgAmount5"] > 0:
            volume_score = min(float(row["AvgAmount5"]) / 100000, 1.0)
        performance = max(min(pct5, 0.20), -0.10) + max(min(pct20, 0.30), -0.15) * 0.5 + volume_score * 0.2
        rows.append({"SCode": symbol, "PerformanceScore": performance})
    return pd.DataFrame(rows, columns=["SCode", "PerformanceScore"])


def compute_news_hot_selection(
    daily_df: pd.DataFrame,
    news_df: pd.DataFrame,
    trade_date: date,
    min_recommendations: int = 3,
    max_recommendations: int = 5,
) -> pd.DataFrame:
    columns = ["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"]
    if daily_df.empty or news_df.empty:
        return pd.DataFrame(columns=columns)
    strategy_df = compute_strategy_frame(daily_df)
    current = strategy_df[strategy_df["TradeDate"] == trade_date].copy()
    if current.empty:
        return pd.DataFrame(columns=columns)
    stocks = current[["SCode", "SName"]].drop_duplicates()
    concept_heat = concept_heat_from_news(news_df)
    news_scores = build_stock_news_scores(news_df, stocks, concept_heat)
    performance_scores = compute_performance_score(strategy_df, trade_date)
    if news_scores.empty:
        return pd.DataFrame(columns=columns)
    selected = current.merge(news_scores, on="SCode", how="inner").merge(performance_scores, on="SCode", how="left")
    selected["PerformanceScore"] = selected["PerformanceScore"].fillna(0)
    selected["Score"] = selected["NewsScore"] * 0.65 + selected["PerformanceScore"] * 0.35
    selected["Reason"] = "news_hot; concepts=" + selected["HotConcepts"].fillna("")
    selected["StrategyName"] = STRATEGY_NEWS_HOT
    selected = selected.sort_values(["Score", "Amount"], ascending=[False, False])
    top_n = max(min_recommendations, min(max_recommendations, len(selected)))
    return selected.head(top_n)[columns]


def compute_weekly_volume_drop_selection(
    weekly_df: pd.DataFrame,
    trade_date: date,
    volume_multiplier: float = 1.5,
    drop_threshold: float = 0.15,
) -> pd.DataFrame:
    columns = ["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"]
    if weekly_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for symbol, group in weekly_df.groupby("SCode", sort=False):
        frame = group.sort_values("TradeDate").reset_index(drop=True)
        eligible = frame[frame["TradeDate"] <= trade_date]
        if len(eligible) < 8:
            continue
        pos = int(eligible.index[-1])
        if pos < 7:
            continue

        first_drop_week = frame.iloc[pos - 1]
        second_drop_week = frame.iloc[pos]
        before_drop_week = frame.iloc[pos - 2]
        prior_five = frame.iloc[pos - 7 : pos - 2]
        if len(prior_five) < 5:
            continue

        close_before = before_drop_week["Close"]
        first_close = first_drop_week["Close"]
        second_close = second_drop_week["Close"]
        prior_volume_avg = prior_five["Volume"].mean(skipna=True)
        drop_volume_avg = pd.Series([first_drop_week["Volume"], second_drop_week["Volume"]]).mean(skipna=True)
        if pd.isna(close_before) or pd.isna(first_close) or pd.isna(second_close) or close_before <= 0:
            continue
        if pd.isna(prior_volume_avg) or prior_volume_avg <= 0 or pd.isna(drop_volume_avg):
            continue

        consecutive_down = first_close < close_before and second_close < first_close
        volume_expanded = drop_volume_avg >= prior_volume_avg * volume_multiplier
        drop_rate = second_close / close_before - 1
        big_drop = drop_rate <= -drop_threshold
        if consecutive_down and volume_expanded and big_drop:
            volume_ratio = float(drop_volume_avg / prior_volume_avg)
            score = abs(float(drop_rate)) * 0.7 + min(volume_ratio / volume_multiplier, 3.0) * 0.3
            rows.append(
                {
                    "TradeDate": second_drop_week["TradeDate"],
                    "SCode": symbol,
                    "SName": second_drop_week["SName"],
                    "Close": second_close,
                    "Score": score,
                    "Reason": f"weekly_volume_drop; drop={drop_rate:.2%}; volume_ratio={volume_ratio:.2f}",
                    "StrategyName": STRATEGY_WEEKLY_VOLUME_DROP,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["Score", "SCode"], ascending=[False, True]).reset_index(drop=True)


def select_stocks_for_date(
    conn: pymysql.connections.Connection,
    trade_date: date,
    min_turnover_amount: float = 0.0,
    limit: int | None = None,
    ktype: str = DEFAULT_KTYPE,
    strategy: str = STRATEGY_MA_BULLISH,
    min_recommendations: int = 3,
    max_recommendations: int = 5,
) -> pd.DataFrame:
    if strategy == STRATEGY_WEEKLY_VOLUME_DROP:
        weekly_df = load_weekly_window(conn, trade_date)
        selected = compute_weekly_volume_drop_selection(weekly_df, trade_date)
        if limit is not None and limit > 0:
            selected = selected.head(limit)
        return selected

    daily_df = load_daily_window(conn, trade_date, DEFAULT_LOOKBACK_DAYS, ktype)
    if strategy == STRATEGY_NEWS_HOT:
        news_df = load_news_for_date(conn, trade_date)
        selected = compute_news_hot_selection(
            daily_df=daily_df,
            news_df=news_df,
            trade_date=trade_date,
            min_recommendations=min_recommendations,
            max_recommendations=max_recommendations,
        )
        if limit is not None and limit > 0:
            selected = selected.head(limit)
        return selected
    if strategy != STRATEGY_MA_BULLISH:
        raise ValueError(f"unsupported strategy: {strategy}")
    strategy_df = compute_strategy_frame(daily_df, min_turnover_amount=min_turnover_amount)
    if strategy_df.empty:
        return pd.DataFrame()
    selected = strategy_df[(strategy_df["TradeDate"] == trade_date) & strategy_df["Selected"]].copy()
    selected = selected.sort_values(["Score", "Amount"], ascending=[False, False])
    if limit is not None and limit > 0:
        selected = selected.head(limit)
    selected["StrategyName"] = STRATEGY_MA_BULLISH
    return selected[["TradeDate", "SCode", "SName", "Close", "Score", "Reason", "StrategyName"]]


def save_selections(conn: pymysql.connections.Connection, selected: pd.DataFrame) -> int:
    ensure_selection_table(conn)
    if selected.empty:
        return 0
    now = datetime.now()
    rows = []
    for row in selected.itertuples(index=False):
        rows.append((row.TradeDate, row.SCode, none_if_nan(row.SName), row.StrategyName, none_if_nan(row.Close), none_if_nan(row.Score), none_if_nan(row.Reason), now))
    sql = f"""
        INSERT INTO {SELECTION_TABLE}
            (TradeDate, SCode, SName, StrategyName, ClosePrice, Score, Reason, CreatedOn)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            SName = VALUES(SName),
            ClosePrice = VALUES(ClosePrice),
            Score = VALUES(Score),
            Reason = VALUES(Reason),
            CreatedOn = VALUES(CreatedOn)
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def run_selection(config: SelectionConfig) -> pd.DataFrame:
    with mysql_connect() as conn:
        trade_date = parse_date(config.trade_date) or latest_trade_date(conn, config.ktype)
        selected = select_stocks_for_date(
            conn=conn,
            trade_date=trade_date,
            min_turnover_amount=config.min_turnover_amount,
            limit=config.limit,
            ktype=config.ktype,
            strategy=config.strategy,
            min_recommendations=config.min_recommendations,
            max_recommendations=config.max_recommendations,
        )
        saved = save_selections(conn, selected)
    print(f"trade_date={trade_date} selected={len(selected)} saved={saved}")
    if not selected.empty:
        print(selected.to_string(index=False))
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run A-share stock selection strategy")
    parser.add_argument("--date", dest="trade_date", help="Selection date, YYYYMMDD or YYYY-MM-DD; default latest trading day in dkandles")
    parser.add_argument("--strategy", choices=STRATEGIES, default=STRATEGY_MA_BULLISH, help="Stock selection strategy")
    parser.add_argument("--min-turnover-amount", type=float, default=0.0, help="Minimum 5-day average Amount; Amount follows dkandles unit")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of selected stocks to save")
    parser.add_argument("--min-recommendations", type=int, default=3, help="Minimum recommendations for news_hot_v1")
    parser.add_argument("--max-recommendations", type=int, default=5, help="Maximum recommendations for news_hot_v1")
    parser.add_argument("--ktype", default=DEFAULT_KTYPE)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_selection(
        SelectionConfig(
            trade_date=args.trade_date,
            min_turnover_amount=args.min_turnover_amount,
            limit=args.limit,
            ktype=args.ktype,
            strategy=args.strategy,
            min_recommendations=args.min_recommendations,
            max_recommendations=args.max_recommendations,
        )
    )


if __name__ == "__main__":
    main()
