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


if __name__ == "__main__":
    unittest.main()
