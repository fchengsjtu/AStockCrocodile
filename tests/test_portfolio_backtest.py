import unittest
from datetime import date

import pandas as pd

from portfolio_backtest.common import PortfolioBacktestConfig, buy_shares_for_budget, round_cent, weighted_average_price
from portfolio_backtest.simulator import simulate_portfolio


class PortfolioBacktestTests(unittest.TestCase):
    def test_rounds_buy_shares_to_hundred_lot(self):
        self.assertEqual(buy_shares_for_budget(10.03, 100000), 10000)
        self.assertEqual(buy_shares_for_budget(10.08, 100000), 9900)

    def test_weighted_average_price_uses_amount_volume_units(self):
        row = type("Row", (), {"Amount": 1000.0, "Volume": 10000.0, "High": 11.0, "Low": 9.0, "Close": 10.0})()
        self.assertEqual(weighted_average_price(row), 10.0)

    def test_simulation_buys_next_day_and_respects_t_plus_one(self):
        signals = pd.DataFrame(
            [{"TradeDate": date(2026, 1, 1), "SCode": "000001", "SName": "A", "Close": 10.0, "Score": 1.0, "Reason": "x", "StrategyName": "test"}]
        )
        daily = pd.DataFrame(
            [
                ["000001", "A", date(2026, 1, 1), 10, 10, 10, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 2), 10, 10, 10, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 3), 10, 10, 12, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 4), 10, 10, 10, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 5), 10, 10, 10, 10, 1000, 10000],
            ],
            columns=["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume"],
        )
        config = PortfolioBacktestConfig(start_date=date(2026, 1, 1), end_date=date(2026, 1, 5), strategy_name="test")
        snapshots, holdings, trades = simulate_portfolio(signals, daily, config, verbose=False)

        buy = trades[trades["Side"] == "BUY"].iloc[0]
        sell = trades[trades["Side"] == "SELL"].iloc[0]
        self.assertEqual(buy.TradeDate, date(2026, 1, 2))
        self.assertEqual(sell.TradeDate, date(2026, 1, 3))
        self.assertEqual(sell.Reason, "take_profit_10pct_half")
        self.assertGreater(len(holdings), 0)
        self.assertIn("TradingFee", snapshots.columns)

    def test_stop_loss_has_priority_over_take_profit_on_same_day(self):
        signals = pd.DataFrame(
            [{"TradeDate": date(2026, 1, 1), "SCode": "000001", "SName": "A", "Close": 10.0, "Score": 1.0, "Reason": "x", "StrategyName": "test"}]
        )
        daily = pd.DataFrame(
            [
                ["000001", "A", date(2026, 1, 1), 10, 10, 10, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 2), 10, 10, 10, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 3), 10, 10, 12, 9.6, 1000, 10000],
            ],
            columns=["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume"],
        )
        config = PortfolioBacktestConfig(start_date=date(2026, 1, 1), end_date=date(2026, 1, 3), strategy_name="test")
        _, _, trades = simulate_portfolio(signals, daily, config, verbose=False)

        sell = trades[trades["Side"] == "SELL"].iloc[0]
        self.assertEqual(sell.Reason, "stop_loss_3pct")
        self.assertEqual(round_cent(sell.Price), 9.70)

    def test_day_three_exit_sells_remaining_position(self):
        signals = pd.DataFrame(
            [{"TradeDate": date(2026, 1, 1), "SCode": "000001", "SName": "A", "Close": 10.0, "Score": 1.0, "Reason": "x", "StrategyName": "test"}]
        )
        daily = pd.DataFrame(
            [
                ["000001", "A", date(2026, 1, 1), 10, 10, 10, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 2), 10, 10, 10, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 3), 10, 10, 10.5, 9.8, 1000, 10000],
                ["000001", "A", date(2026, 1, 4), 10, 10, 10.5, 9.8, 1000, 10000],
                ["000001", "A", date(2026, 1, 5), 10, 10.2, 10.5, 9.8, 1000, 10000],
            ],
            columns=["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume"],
        )
        config = PortfolioBacktestConfig(start_date=date(2026, 1, 1), end_date=date(2026, 1, 5), strategy_name="test")
        _, _, trades = simulate_portfolio(signals, daily, config, verbose=False)

        sell = trades[trades["Side"] == "SELL"].iloc[0]
        self.assertEqual(sell.TradeDate, date(2026, 1, 5))
        self.assertEqual(sell.Reason, "time_exit_day3_close")


if __name__ == "__main__":
    unittest.main()
