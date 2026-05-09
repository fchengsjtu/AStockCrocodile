from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Callable, Iterable
from urllib.parse import urljoin

import akshare as ak
import pandas as pd
import pymysql
import requests

from a_share_crawler import mysql_connect, none_if_nan

NEWS_TABLE = "news"
DEFAULT_SOURCES = (
    "eastmoney",
    "ths",
    "caixin",
    "yicai",
    "eeo",
    "21jingji",
    "caijing",
    "ce",
    "jwview",
    "stcn",
    "cnstock",
    "sina",
    "xueqiu",
    "jiemian",
    "hexun",
    "stockstar",
)
DEFAULT_NEWS_LIMIT = 10000
MAX_RELATED_CONCEPTS = 10
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SOURCE_CREDIBILITY = {
    "eastmoney": 3,
    "ths": 4,
    "caixin": 2,
    "yicai": 2,
    "eeo": 2,
    "21jingji": 2,
    "caijing": 2,
    "ce": 3,
    "jwview": 3,
    "stcn": 2,
    "cnstock": 2,
    "sina": 4,
    "xueqiu": 6,
    "jiemian": 3,
    "hexun": 4,
    "stockstar": 4,
}

WEB_SOURCE_CONFIGS = {
    "yicai": {
        "url": "https://www.yicai.com/",
        "base_url": "https://www.yicai.com/",
        "mode": "html",
    },
    "eeo": {
        "url": "https://www.eeo.com.cn/finance/rss.xml",
        "base_url": "https://www.eeo.com.cn/",
        "mode": "rss",
    },
    "21jingji": {
        "url": "https://www.21jingji.com/",
        "base_url": "https://www.21jingji.com/",
        "mode": "html",
    },
    "caijing": {
        "url": "https://www.caijing.com.cn/",
        "base_url": "https://www.caijing.com.cn/",
        "mode": "html",
    },
    "ce": {
        "url": "http://www.ce.cn/",
        "base_url": "http://www.ce.cn/",
        "mode": "html",
    },
    "jwview": {
        "url": "https://www.jwview.com/",
        "base_url": "https://www.jwview.com/",
        "mode": "html",
    },
    "stcn": {
        "url": "https://www.stcn.com/",
        "base_url": "https://www.stcn.com/",
        "mode": "html",
    },
    "cnstock": {
        "url": "https://www.cnstock.com/",
        "base_url": "https://www.cnstock.com/",
        "mode": "html",
    },
    "sina": {
        "url": "https://finance.sina.com.cn/",
        "base_url": "https://finance.sina.com.cn/",
        "mode": "html",
    },
    "xueqiu": {
        "url": "https://xueqiu.com/",
        "base_url": "https://xueqiu.com/",
        "mode": "html",
    },
    "jiemian": {
        "url": "https://www.jiemian.com/lists/48.html",
        "base_url": "https://www.jiemian.com/",
        "mode": "html",
    },
    "hexun": {
        "url": "https://www.hexun.com/",
        "base_url": "https://www.hexun.com/",
        "mode": "html",
    },
    "stockstar": {
        "url": "https://www.stockstar.com/",
        "base_url": "https://www.stockstar.com/",
        "mode": "html",
    },
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


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def decode_response_text(response: requests.Response) -> str:
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def fetch_text(url: str) -> str:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()
    return decode_response_text(response)


def normalize_publish_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).replace(tzinfo=None)
    except (TypeError, ValueError, AttributeError):
        return normalize_time(value)


def rows_to_news_frame(rows: list[dict], source: str) -> pd.DataFrame:
    if not rows:
        return normalize_news_frame(pd.DataFrame(), source, {"link": "NewsLink"})
    df = pd.DataFrame(rows)
    return normalize_news_frame(
        df,
        source,
        {"title": "Title", "summary": "Summary", "publish_time": "PublishTime", "link": "NewsLink", "heat": "Heat"},
    )


def fetch_rss_source(source: str, url: str) -> pd.DataFrame:
    text = fetch_text(url)
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text, count=1)
    root = ET.fromstring(text)
    rows = []
    for item in root.findall(".//item"):
        title = clean_text(item.findtext("title"))
        link = clean_text(item.findtext("link"))
        summary = clean_text(item.findtext("description"))
        publish_time = normalize_publish_time(item.findtext("pubDate") or item.findtext("date"))
        if not link:
            continue
        rows.append(
            {
                "Title": title,
                "Summary": summary,
                "PublishTime": publish_time,
                "NewsLink": link,
                "Heat": 0,
            }
        )
    return rows_to_news_frame(rows, source)


def fetch_html_source(source: str, url: str, base_url: str) -> pd.DataFrame:
    text = fetch_text(url)
    rows = []
    seen = set()
    pattern = re.compile(r"<a\b[^>]*?href=[\"'](?P<href>[^\"'#]+)[\"'][^>]*>(?P<title>.*?)</a>", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(text):
        href = html.unescape(match.group("href")).strip()
        title = clean_text(match.group("title"))
        if len(title) < 8:
            continue
        if href.startswith(("javascript:", "mailto:")):
            continue
        link = urljoin(base_url, href)
        if not link.startswith(("http://", "https://")) or link in seen:
            continue
        seen.add(link)
        rows.append(
            {
                "Title": title[:512],
                "Summary": title,
                "PublishTime": None,
                "NewsLink": link,
                "Heat": 0,
            }
        )
        if len(rows) >= 300:
            break
    return rows_to_news_frame(rows, source)


def fetch_web_source(source: str) -> pd.DataFrame:
    config = WEB_SOURCE_CONFIGS[source]
    if config["mode"] == "rss":
        return fetch_rss_source(source, config["url"])
    return fetch_html_source(source, config["url"], config["base_url"])


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

for _source_name in WEB_SOURCE_CONFIGS:
    FETCHERS[_source_name] = lambda source_name=_source_name: fetch_web_source(source_name)


def crawl_news(sources: Iterable[str], limit: int) -> pd.DataFrame:
    frames = []
    for source in sources:
        if source not in FETCHERS:
            raise ValueError(f"unsupported news source: {source}")
        try:
            frame = FETCHERS[source]()
        except Exception as exc:
            print(f"WARNING source={source} failed: {exc}")
            continue
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
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="Comma-separated sources; default includes all configured sources")
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
