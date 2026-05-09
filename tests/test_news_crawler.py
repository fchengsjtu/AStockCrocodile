import json
import unittest

import pandas as pd

import news_crawler


class NewsCrawlerTests(unittest.TestCase):
    def test_score_related_concepts_orders_by_weight(self):
        concepts = news_crawler.score_related_concepts(
            "AI大模型带动算力和芯片需求",
            "人工智能公司采购半导体芯片，云计算和软件需求提升",
        )

        self.assertEqual(concepts[0], "AI")
        self.assertIn("半导体", concepts)
        self.assertLessEqual(len(concepts), 10)

    def test_normalize_heat_supports_wan_suffix(self):
        self.assertEqual(news_crawler.normalize_heat("1.5万"), 15000)
        self.assertEqual(news_crawler.normalize_heat("2,300"), 2300)
        self.assertEqual(news_crawler.normalize_heat(""), 0)

    def test_normalize_news_frame_maps_required_fields(self):
        df = pd.DataFrame(
            [
                {
                    "标题": "新能源汽车产业链订单增长",
                    "摘要": "动力电池和储能需求改善",
                    "发布时间": "2026-05-09 10:30:00",
                    "链接": "https://example.com/news/1",
                    "点击": "1.2万",
                }
            ]
        )

        result = news_crawler.normalize_news_frame(
            df,
            "eastmoney",
            {"title": "标题", "summary": "摘要", "publish_time": "发布时间", "link": "链接", "heat": "点击"},
        )

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["NewsLink"], "https://example.com/news/1")
        self.assertEqual(row["CredibilityLevel"], 3)
        self.assertEqual(row["Heat"], 12000)
        related = json.loads(row["RelatedConcepts"])
        self.assertIn("新能源车", related)
        self.assertIn("锂电池", related)
        self.assertTrue(row["ContentHash"])

    def test_crawl_news_deduplicates_and_limits(self):
        original_fetchers = news_crawler.FETCHERS
        try:
            news_crawler.FETCHERS = {
                "one": lambda: pd.DataFrame(
                    [
                        {
                            "NewsLink": "https://example.com/a",
                            "Title": "A",
                            "Summary": "",
                            "SourceName": "one",
                            "PublishTime": pd.Timestamp("2026-05-09 10:00:00"),
                            "CredibilityLevel": 5,
                            "Heat": 1,
                            "RelatedConcepts": "[]",
                            "ContentHash": "a",
                        }
                    ]
                ),
                "two": lambda: pd.DataFrame(
                    [
                        {
                            "NewsLink": "https://example.com/a",
                            "Title": "A duplicate",
                            "Summary": "",
                            "SourceName": "two",
                            "PublishTime": pd.Timestamp("2026-05-09 11:00:00"),
                            "CredibilityLevel": 5,
                            "Heat": 1,
                            "RelatedConcepts": "[]",
                            "ContentHash": "b",
                        },
                        {
                            "NewsLink": "https://example.com/b",
                            "Title": "B",
                            "Summary": "",
                            "SourceName": "two",
                            "PublishTime": pd.Timestamp("2026-05-09 12:00:00"),
                            "CredibilityLevel": 5,
                            "Heat": 1,
                            "RelatedConcepts": "[]",
                            "ContentHash": "c",
                        },
                    ]
                ),
            }

            result = news_crawler.crawl_news(("one", "two"), limit=1)

            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["NewsLink"], "https://example.com/b")
        finally:
            news_crawler.FETCHERS = original_fetchers


if __name__ == "__main__":
    unittest.main()
