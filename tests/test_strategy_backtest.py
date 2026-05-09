import unittest
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd

import backtest_strategy as backtest
import stock_selector


class StrategyBacktestTests(unittest.TestCase):
    def test_weighted_average_price_uses_amount_volume_units(self):
        window = pd.DataFrame(
            {
                "Amount": [1000.0, 2000.0],
                "Volume": [10000.0, 10000.0],
                "High": [11.0, 12.0],
                "Low": [9.0, 10.0],
                "Close": [10.0, 11.0],
            }
        )

        self.assertEqual(backtest.weighted_average_price(window), 15.0)

    def test_evaluate_selection_uses_third_to_eighth_trading_days(self):
        base = date(2026, 1, 1)
        rows = []
        for i in range(10):
            rows.append(
                {
                    "SCode": "000001",
                    "TradeDate": base + timedelta(days=i),
                    "Low": 10.0 if i < 3 else 10.3,
                    "High": 11.0,
                    "Close": 10.0,
                    "Amount": 1000.0,
                    "Volume": 10000.0,
                }
            )
        frames = backtest.build_trade_day_positions(pd.DataFrame(rows))
        selection = SimpleNamespace(SCode="000001", SName="Ping An", TradeDate=base, Close=10.0, Score=1.0, Reason="test")

        result = backtest.evaluate_selection(selection, frames)

        self.assertEqual(result["ForwardStart"], base + timedelta(days=3))
        self.assertEqual(result["ForwardEnd"], base + timedelta(days=8))
        self.assertTrue(result["Success"])
        self.assertFalse(result["Explosive"])
        self.assertAlmostEqual(result["RiseRate"], 0.03)

    def test_evaluate_selection_failure_and_explosive(self):
        base = date(2026, 1, 1)
        explosive_rows = []
        for i in range(10):
            explosive_rows.append(
                {
                    "SCode": "000001",
                    "TradeDate": base + timedelta(days=i),
                    "Low": 12.1 if i >= 3 else 10.0,
                    "High": 13.0,
                    "Close": 12.5,
                    "Amount": 1250.0,
                    "Volume": 10000.0,
                }
            )
        frames = backtest.build_trade_day_positions(pd.DataFrame(explosive_rows))
        selection = SimpleNamespace(SCode="000001", SName="Ping An", TradeDate=base, Close=10.0, Score=1.0, Reason="test")

        explosive = backtest.evaluate_selection(selection, frames)

        self.assertTrue(explosive["Success"])
        self.assertTrue(explosive["Explosive"])

        failure_rows = []
        for i in range(10):
            failure_rows.append(
                {
                    "SCode": "000002",
                    "TradeDate": base + timedelta(days=i),
                    "Low": 9.5,
                    "High": 10.5,
                    "Close": 10.0,
                    "Amount": 980.0 if i >= 3 else 1000.0,
                    "Volume": 10000.0,
                }
            )
        frames = backtest.build_trade_day_positions(pd.DataFrame(failure_rows))
        selection = SimpleNamespace(SCode="000002", SName="Vanke", TradeDate=base, Close=10.0, Score=1.0, Reason="test")

        failure = backtest.evaluate_selection(selection, frames)

        self.assertTrue(failure["Failure"])
        self.assertAlmostEqual(failure["WeightedDropRate"], -0.02)

    def test_summarize_results_by_stock(self):
        results = pd.DataFrame(
            [
                {"SCode": "000001", "Success": True, "Failure": False, "Explosive": False, "RiseRate": 0.03, "WeightedDropRate": 0.01},
                {"SCode": "000001", "Success": False, "Failure": True, "Explosive": False, "RiseRate": -0.01, "WeightedDropRate": -0.02},
                {"SCode": "000002", "Success": True, "Failure": False, "Explosive": True, "RiseRate": 0.21, "WeightedDropRate": 0.05},
            ]
        )

        summary = backtest.summarize_results_by_stock(results, "test_strategy", date(2026, 1, 1), date(2026, 1, 31))

        first = summary[summary["SCode"] == "000001"].iloc[0]
        self.assertEqual(first["StrategyName"], "test_strategy")
        self.assertEqual(first["SampleCount"], 2)
        self.assertEqual(first["SuccessRate"], 0.5)
        self.assertAlmostEqual(first["AvgRiseRate"], 0.01)
        self.assertEqual(first["FailureRate"], 0.5)
        self.assertAlmostEqual(first["AvgDropRate"], -0.005)
        self.assertEqual(first["ExplosiveRate"], 0.0)

    def test_compute_strategy_frame_marks_default_signal(self):
        base = date(2026, 1, 1)
        rows = []
        for i in range(60):
            close = 10.0 + i * 0.1
            rows.append(
                {
                    "SCode": "000001",
                    "SName": "Ping An",
                    "TradeDate": base + timedelta(days=i),
                    "Open": close - 0.05,
                    "Close": close,
                    "High": close + 0.1,
                    "Low": close - 0.1,
                    "Amount": 1000.0,
                    "Volume": 10000.0,
                    "MA5": close - 0.1,
                    "MA8": close - 0.2,
                    "MA13": close - 0.3,
                    "MA34": close - 0.4,
                    "MA55": close - 0.5,
                }
            )
        df = pd.DataFrame(rows)

        result = stock_selector.compute_strategy_frame(df)

        self.assertTrue(result.iloc[-1]["Selected"])


if __name__ == "__main__":
    unittest.main()
