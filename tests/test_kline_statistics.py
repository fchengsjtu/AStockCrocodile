import unittest
from datetime import date, timedelta

import pandas as pd

import kline_statistics


class KlineStatisticsTests(unittest.TestCase):
    def test_find_short_term_surges_uses_third_future_trading_day(self):
        base = date(2026, 1, 1)
        rows = []
        closes = [9.8, 10.0, 10.5, 11.0, 12.1, 12.0]
        for i, close in enumerate(closes):
            rows.append(
                {
                    "SCode": "000001",
                    "SName": "Ping An",
                    "TradeDate": base + timedelta(days=i),
                    "Close": close,
                }
            )
        df = pd.DataFrame(rows)

        result = kline_statistics.find_short_term_surges(
            daily_df=df,
            start_date=base + timedelta(days=1),
            end_date=base + timedelta(days=1),
            forward_days=3,
            threshold=0.20,
        )

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["SCode"], "000001")
        self.assertEqual(row["StartRiseDate"], base + timedelta(days=1))
        self.assertEqual(row["PrevTradeDate"], base)
        self.assertAlmostEqual(row["GainRate"], 0.21)
        self.assertEqual(row["StatType"], kline_statistics.SHORT_TERM_SURGE_TYPE)

    def test_find_short_term_surges_skips_without_previous_or_forward_day(self):
        base = date(2026, 1, 1)
        df = pd.DataFrame(
            [
                {"SCode": "000001", "SName": "Ping An", "TradeDate": base + timedelta(days=i), "Close": 10.0 + i}
                for i in range(4)
            ]
        )

        result = kline_statistics.find_short_term_surges(
            daily_df=df,
            start_date=base,
            end_date=base + timedelta(days=3),
            forward_days=3,
            threshold=0.20,
        )

        self.assertTrue(result.empty)

    def test_iter_batches_uses_small_chunks(self):
        batches = list(kline_statistics.iter_batches(["000001", "000002", "000003"], 2))

        self.assertEqual(batches, [["000001", "000002"], ["000003"]])
        self.assertLessEqual(kline_statistics.DEFAULT_SYMBOL_BATCH_SIZE, 100)

    def test_load_daily_kline_for_symbols_skips_query_without_symbols(self):
        class FailingConnection:
            def cursor(self):
                raise AssertionError("database should not be queried without symbols")

        result = kline_statistics.load_daily_kline_for_symbols(
            FailingConnection(),
            [],
            date(2026, 1, 1),
            date(2026, 1, 31),
            "D",
            3,
        )

        self.assertTrue(result.empty)

    def test_message_driven_news_requires_stock_mention_and_keyword(self):
        self.assertTrue(
            kline_statistics.has_message_driven_news(
                "000001",
                "平安银行",
                "平安银行公告业绩预增",
                "",
            )
        )
        self.assertFalse(
            kline_statistics.has_message_driven_news(
                "000001",
                "平安银行",
                "银行板块上涨",
                "没有个股消息",
            )
        )
        self.assertFalse(
            kline_statistics.has_message_driven_news(
                "000001",
                "平安银行",
                "某公司公告重大重组",
                "",
            )
        )

    def test_filter_message_driven_surges_excludes_matching_news(self):
        stats = pd.DataFrame(
            [
                {
                    "SCode": "000001",
                    "SName": "平安银行",
                    "StartRiseDate": date(2026, 1, 5),
                    "PrevTradeDate": date(2026, 1, 4),
                    "GainRate": 0.25,
                    "StatType": kline_statistics.SHORT_TERM_SURGE_TYPE,
                },
                {
                    "SCode": "000002",
                    "SName": "万科A",
                    "StartRiseDate": date(2026, 1, 5),
                    "PrevTradeDate": date(2026, 1, 4),
                    "GainRate": 0.22,
                    "StatType": kline_statistics.SHORT_TERM_SURGE_TYPE,
                },
            ]
        )
        news = pd.DataFrame(
            [
                {
                    "PublishDate": date(2026, 1, 6),
                    "Title": "平安银行公告重大合同",
                    "Summary": "",
                }
            ]
        )

        filtered, excluded = kline_statistics.filter_message_driven_surges(stats, news, 3)

        self.assertEqual(excluded, 1)
        self.assertEqual(filtered["SCode"].tolist(), ["000002"])


if __name__ == "__main__":
    unittest.main()
