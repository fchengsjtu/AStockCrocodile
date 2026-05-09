from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

import akshare as ak
import pandas as pd
import pymysql

from a_share_crawler import mysql_connect, none_if_nan

NEWS_TABLE = "news"
DEFAULT_SOURCES = ("eastmoney", "ths", "caixin")
DEFAULT_NEWS_LIMIT = 10000
MAX_RELATED_CONCEPTS = 10

SOURCE_CREDIBILITY = {
    "eastmoney": 3,
    "ths": 4,
    "caixin": 2,
}

CONCEPT_KEYWORDS = {
    "AI": ("人工智能", "AI", "大模型", "算力", "机器人", "智能体", "AIGC"),
    "半导体": ("半导体", "芯片", "集成电路", "晶圆", "封测", "光刻"),
    "新能源车": ("新能源汽车", "新能源车", "电动车", "智能驾驶", "汽车", "车企"),
    "锂电池": ("锂电", "电池", "动力电池", "固态电池", "储能电池"),
    "光伏": ("光伏", "硅片", "组件", "逆变器", "太阳能"),
    "储能": ("储能", "新型储能", "电网侧", "电力系统"),
    "低空经济": ("低空经济", "eVTOL", "飞行汽车", "无人机"),
    "机器人": ("机器人", "人形机器人", "工业机器人", "减速器"),
    "医药": ("医药", "创新药", "医疗", "生物医药", "药企", "CXO"),
    "银行": ("银行", "贷款", "存款", "息差", "信贷"),
    "证券": ("证券", "券商", "资本市场", "投行", "两融"),
    "保险": ("保险", "保费", "险资", "寿险", "财险"),
    "房地产": ("房地产", "地产", "楼市", "房企", "商品房"),
    "消费": ("消费", "零售", "食品饮料", "白酒", "旅游", "餐饮"),
    "军工": ("军工", "航空发动机", "导弹", "卫星", "国防"),
    "煤炭": ("煤炭", "煤价", "焦煤", "焦炭"),
    "有色金属": ("有色", "铜", "铝", "黄金", "稀土", "金属"),
    "化工": ("化工", "化肥", "磷化工", "煤化工", "纯碱"),
    "数字经济": ("数字经济", "数据要素", "云计算", "信创", "软件"),
    "一带一路": ("一带一路", "出海", "海外订单", "外贸", "出口"),
}


@dataclass(frozen=True)
class NewsConfig:
    sources: tuple[str, ...]
    limit: int
    save_db: bool
    output: str | None


def ensure_news_table(conn: pymysql.connections.Connection) -> None:
    sql = f"""
        CREATE TABLE IF NOT EXISTS {NEWS_TABLE} (
            Id BIGINT NOT NULL AUTO_INCREMENT,
            NewsLink VARCHAR(512) NOT NULL,
            Title VARCHAR(512) NULL,
            Summary TEXT NULL,
            SourceName VARCHAR(64) NOT NULL,
            PublishTime DATETIME NULL,
            CredibilityLevel TINYINT NOT NULL,
            Heat BIGINT NOT NULL DEFAULT 0,
            RelatedConcepts JSON NULL,
            ConceptHeat JSON NULL,
            ContentHash CHAR(64) NOT NULL DEFAULT '',
            CreatedOn DATETIME NOT NULL,
            UpdatedOn DATETIME NOT NULL,
            PRIMARY KEY (Id),
            UNIQUE KEY ux_news_link (NewsLink),
            KEY idx_news_source_time (SourceName, PublishTime),
            KEY idx_news_credibility (CredibilityLevel)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute("SHOW COLUMNS FROM news LIKE 'ConceptHeat'")
        if cur.fetchone() is None:
            cur.execute("ALTER TABLE news ADD COLUMN ConceptHeat JSON NULL AFTER RelatedConcepts")
    conn.commit()


def normalize_time(value) -> datetime | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def normalize_heat(value) -> int:
    if value is None or pd.isna(value):
        return 0
    text = str(value).strip().replace(",", "")
    multiplier = 1
    if text.endswith("万"):
        multiplier = 10000
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return 0


def score_related_concepts(title: str | None, summary: str | None, limit: int = MAX_RELATED_CONCEPTS) -> list[str]:
    title_text = title or ""
    summary_text = summary or ""
    scores = []
    for concept, keywords in CONCEPT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            score += title_text.count(keyword) * 3
            score += summary_text.count(keyword)
        if score > 0:
            scores.append((concept, score))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return [concept for concept, _ in scores[:limit]]


def content_hash(row: dict) -> str:
    payload = "|".join(
        str(row.get(key) or "")
        for key in ("NewsLink", "Title", "Summary", "SourceName", "PublishTime", "CredibilityLevel", "Heat", "RelatedConcepts", "ConceptHeat")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_json_list(value) -> list:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def compute_concept_heat(news_df: pd.DataFrame) -> dict[str, float]:
    if news_df.empty or "RelatedConcepts" not in news_df.columns:
        return {}
    total = len(news_df)
    counts: dict[str, int] = {}
    for value in news_df["RelatedConcepts"]:
        concepts = parse_json_list(value)
        for concept in set(str(item) for item in concepts[:MAX_RELATED_CONCEPTS] if item):
            counts[concept] = counts.get(concept, 0) + 1
    if total == 0:
        return {}
    return {concept: count / total for concept, count in counts.items()}


def apply_concept_heat(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return news_df
    result = news_df.copy()
    heat_map = compute_concept_heat(result)
    values = []
    hashes = []
    for row in result.itertuples(index=False):
        concepts = parse_json_list(getattr(row, "RelatedConcepts", None))
        concept_heat = [
            {"concept": concept, "heat": heat_map.get(str(concept), 0.0)}
            for concept in concepts[:MAX_RELATED_CONCEPTS]
        ]
        values.append(json.dumps(concept_heat, ensure_ascii=False))
    result["ConceptHeat"] = values
    for row in result.to_dict("records"):
        hashes.append(content_hash(row))
    result["ContentHash"] = hashes
    return result


def normalize_news_frame(df: pd.DataFrame, source: str, mapping: dict[str, str]) -> pd.DataFrame:
    rows = []
    for _, item in df.iterrows():
        link = item.get(mapping["link"])
        if link is None or pd.isna(link) or not str(link).strip():
            continue
        title = item.get(mapping.get("title", ""), "")
        summary = item.get(mapping.get("summary", ""), "")
        if summary is None or pd.isna(summary):
            summary = ""
        publish_time = normalize_time(item.get(mapping.get("publish_time", ""), None))
        heat = normalize_heat(item.get(mapping.get("heat", ""), 0))
        related = score_related_concepts(str(title or ""), str(summary or ""))
        row = {
            "NewsLink": str(link).strip(),
            "Title": str(title or "").strip() or None,
            "Summary": str(summary or "").strip() or None,
            "SourceName": source,
            "PublishTime": publish_time,
            "CredibilityLevel": SOURCE_CREDIBILITY.get(source, 5),
            "Heat": heat,
            "RelatedConcepts": json.dumps(related, ensure_ascii=False),
            "ConceptHeat": None,
        }
        row["ContentHash"] = content_hash(row)
        rows.append(row)
    columns = ["NewsLink", "Title", "Summary", "SourceName", "PublishTime", "CredibilityLevel", "Heat", "RelatedConcepts", "ConceptHeat", "ContentHash"]
    return pd.DataFrame(rows, columns=columns)


def fetch_eastmoney_news() -> pd.DataFrame:
    df = ak.stock_info_global_em()
    return normalize_news_frame(
        df,
        "eastmoney",
        {"title": "标题", "summary": "摘要", "publish_time": "发布时间", "link": "链接"},
    )


def fetch_ths_news() -> pd.DataFrame:
    df = ak.stock_info_global_ths()
    return normalize_news_frame(
        df,
        "ths",
        {"title": "标题", "summary": "内容", "publish_time": "发布时间", "link": "链接"},
    )


def fetch_caixin_news() -> pd.DataFrame:
    df = ak.stock_news_main_cx()
    return normalize_news_frame(
        df,
        "caixin",
        {"title": "tag", "summary": "summary", "link": "url"},
    )


FETCHERS: dict[str, Callable[[], pd.DataFrame]] = {
    "eastmoney": fetch_eastmoney_news,
    "ths": fetch_ths_news,
    "caixin": fetch_caixin_news,
}


def crawl_news(sources: Iterable[str], limit: int) -> pd.DataFrame:
    frames = []
    for source in sources:
        if source not in FETCHERS:
            raise ValueError(f"unsupported news source: {source}")
        frame = FETCHERS[source]()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["NewsLink", "Title", "Summary", "SourceName", "PublishTime", "CredibilityLevel", "Heat", "RelatedConcepts", "ConceptHeat", "ContentHash"])
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["NewsLink"], keep="first")
    result = result.sort_values(["PublishTime", "Heat"], ascending=[False, False], na_position="last")
    if limit > 0:
        result = result.head(limit)
    return apply_concept_heat(result.reset_index(drop=True))


def save_news(conn: pymysql.connections.Connection, news_df: pd.DataFrame) -> int:
    ensure_news_table(conn)
    if news_df.empty:
        return 0
    now = datetime.now()
    rows = []
    for row in news_df.itertuples(index=False):
        rows.append(
            (
                row.NewsLink,
                none_if_nan(row.Title),
                none_if_nan(row.Summary),
                row.SourceName,
                none_if_nan(row.PublishTime),
                row.CredibilityLevel,
                row.Heat,
                row.RelatedConcepts,
                row.ConceptHeat,
                row.ContentHash,
                now,
                now,
            )
        )
    sql = f"""
        INSERT INTO {NEWS_TABLE}
            (NewsLink, Title, Summary, SourceName, PublishTime, CredibilityLevel, Heat, RelatedConcepts, ConceptHeat, ContentHash, CreatedOn, UpdatedOn)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            Title = VALUES(Title),
            Summary = VALUES(Summary),
            SourceName = VALUES(SourceName),
            PublishTime = VALUES(PublishTime),
            CredibilityLevel = VALUES(CredibilityLevel),
            Heat = VALUES(Heat),
            RelatedConcepts = VALUES(RelatedConcepts),
            ConceptHeat = VALUES(ConceptHeat),
            ContentHash = VALUES(ContentHash),
            UpdatedOn = VALUES(UpdatedOn)
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def run_news_crawler(config: NewsConfig) -> pd.DataFrame:
    news_df = crawl_news(config.sources, config.limit)
    if config.output:
        news_df.to_csv(config.output, index=False, encoding="utf-8-sig")
    saved = 0
    if config.save_db:
        with mysql_connect() as conn:
            saved = save_news(conn, news_df)
    print(f"sources={','.join(config.sources)} fetched={len(news_df)} saved={saved}")
    return news_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crawl stock market news into MySQL")
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="Comma-separated sources: eastmoney,ths,caixin")
    parser.add_argument("--limit", type=int, default=DEFAULT_NEWS_LIMIT, help="Maximum news rows to keep after de-duplication; <=0 means no limit")
    parser.add_argument("--output", help="Optional CSV path for crawled news rows")
    parser.add_argument("--no-save-db", action="store_true", help="Do not save news rows to MySQL")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sources = tuple(item.strip() for item in args.sources.split(",") if item.strip())
    run_news_crawler(
        NewsConfig(
            sources=sources,
            limit=args.limit,
            save_db=not args.no_save_db,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
