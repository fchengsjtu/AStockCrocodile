import unittest
from datetime import date, timedelta

import pandas as pd

import surge_pattern_miner


class SurgePatternMinerTests(unittest.TestCase):
    def _daily_window(self):
        base = date(2026, 1, 1)
        rows = []
        for i in range(60):
            close = 10 + i * 0.1
            rows.append(
                {
                    "SCode": "000001",
                    "SName": "Ping An",
                    "TradeDate": base + timedelta(days=i),
                    "Open": close - 0.05,
                    "Close": close,
                    "High": close + 0.1,
                    "Low": close - 0.1,
                    "Volume": 1000 + i * 10,
                    "Amount": 100000 + i * 1000,
                    "MA5": close - 0.1,
                    "MA13": close - 0.2,
                    "MA34": close - 0.3,
                    "MA55": close - 0.4,
                }
            )
        return pd.DataFrame(rows)

    def _weekly_window(self):
        base = date(2025, 1, 3)
        rows = []
        for i in range(60):
            close = 8 + i * 0.2
            rows.append(
                {
                    "SCode": "000001",
                    "SName": "Ping An",
                    "TradeDate": base + timedelta(days=i * 7),
                    "Open": close - 0.1,
                    "Close": close,
                    "High": close + 0.2,
                    "Low": close - 0.2,
                    "Volume": 2000 + i * 20,
                    "Amount": 200000 + i * 2000,
                    "MA5": close - 0.1,
                    "MA13": close - 0.2,
                    "MA34": close - 0.3,
                    "MA55": close - 0.4,
                }
            )
        return pd.DataFrame(rows)

    def test_build_pattern_features_uses_daily_and_weekly_windows(self):
        features = surge_pattern_miner.build_pattern_features(self._daily_window().tail(55), self._weekly_window().tail(55))

        self.assertIn("D_CLOSE_GT_MA5", features)
        self.assertIn("W_CLOSE_GT_MA5", features)
        self.assertIn("D_MA5_GT_MA13", features)
        self.assertIn("W_MA34_GT_MA55", features)

    def test_extract_features_requires_full_windows(self):
        daily = self._daily_window().head(20)
        weekly = self._weekly_window().head(20)

        features = surge_pattern_miner.extract_features_for_date(
            daily,
            weekly,
            daily.iloc[-1]["TradeDate"],
            daily_window_size=55,
            weekly_window_size=55,
        )

        self.assertIsNone(features)

    def test_iter_pattern_keys_returns_sorted_combinations(self):
        patterns = list(surge_pattern_miner.iter_pattern_keys({"B", "A", "C"}, 2))

        self.assertIn(("A",), patterns)
        self.assertIn(("A", "B"), patterns)
        self.assertIn(("B", "C"), patterns)
        self.assertNotIn(("A", "B", "C"), patterns)


if __name__ == "__main__":
    unittest.main()
