import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from portfolio_backtest.common import BLACKBOX_STRATEGIES, DEFAULT_FEE_RATE, DEFAULT_STAMP_DUTY_RATE, STOP_LOSS_RULE_NAMES, STOP_LOSS_SERIES, PortfolioBacktestConfig, buy_shares_for_budget, filter_selection_candidates, is_special_treatment_stock_name, round_cent, stop_loss_pct_from_rule_name, stop_loss_rule_name, weighted_average_price
from portfolio_backtest import db as portfolio_db
from portfolio_backtest import pool_run
from portfolio_backtest import run as portfolio_run
from portfolio_backtest.simulator import simulate_portfolio


class PortfolioBacktestTests(unittest.TestCase):
    def test_default_fee_rate_is_two_basis_points(self):
        self.assertEqual(DEFAULT_FEE_RATE, 0.0002)
        self.assertEqual(portfolio_run.build_parser().parse_args([]).fee_rate, 0.0002)
        pool_args = pool_run.build_parser().parse_args(
            ["--pool-strategy-name", "blackbox_finetune_recall60"]
        )
        self.assertEqual(pool_args.fee_rate, 0.0002)

    def test_default_stamp_duty_is_five_basis_points_on_sells(self):
        self.assertEqual(DEFAULT_STAMP_DUTY_RATE, 0.0005)
        self.assertEqual(portfolio_run.build_parser().parse_args([]).stamp_duty_rate, 0.0005)
        pool_args = pool_run.build_parser().parse_args(
            ["--pool-strategy-name", "blackbox_finetune_recall60"]
        )
        self.assertEqual(pool_args.stamp_duty_rate, 0.0005)

    def test_special_treatment_names_are_excluded(self):
        self.assertTrue(is_special_treatment_stock_name("*ST示例"))
        self.assertTrue(is_special_treatment_stock_name("PT示例"))
        self.assertFalse(is_special_treatment_stock_name("正常股份"))

    def test_selection_filter_excludes_st_and_applies_thirteen_day_cooldown(self):
        trade_dates = [date(2026, 1, day) for day in range(1, 16)]
        frame = pd.DataFrame(
            [
                {"TradeDate": trade_dates[0], "SCode": "1", "SName": "正常股份", "Score": 0.9},
                {"TradeDate": trade_dates[1], "SCode": "2", "SName": "*ST风险", "Score": 1.0},
                {"TradeDate": trade_dates[13], "SCode": "1", "SName": "正常股份", "Score": 0.95},
                {"TradeDate": trade_dates[14], "SCode": "1", "SName": "正常股份", "Score": 0.8},
            ]
        )

        result = filter_selection_candidates(frame, trade_dates, cooldown_trading_days=13)

        self.assertEqual(result["TradeDate"].tolist(), [trade_dates[0], trade_dates[14]])
        self.assertEqual(result["SCode"].tolist(), ["000001", "000001"])

    def test_selection_filter_applies_daily_limit_after_exclusions(self):
        trade_dates = [date(2026, 1, 5)]
        frame = pd.DataFrame(
            [
                {"TradeDate": trade_dates[0], "SCode": "1", "SName": "*ST一", "Score": 1.0},
                {"TradeDate": trade_dates[0], "SCode": "2", "SName": "二", "Score": 0.9},
                {"TradeDate": trade_dates[0], "SCode": "3", "SName": "三", "Score": 0.8},
            ]
        )

        result = filter_selection_candidates(frame, trade_dates, limit_per_day=2)

        self.assertEqual(result["SCode"].tolist(), ["000002", "000003"])

    def test_run_parser_accepts_blackbox_recall_strategies(self):
        for strategy in BLACKBOX_STRATEGIES:
            with self.subTest(strategy=strategy):
                args = portfolio_run.build_parser().parse_args(["--strategy-name", strategy])
                self.assertEqual(args.strategy_name, strategy)

    def test_load_strategy_signals_routes_blackbox_strategy(self):
        expected = pd.DataFrame(
            [{"TradeDate": date(2026, 1, 5), "SCode": "000001", "SName": "A", "Close": 10.0, "Score": 0.9, "Reason": "x", "StrategyName": "blackbox_finetune_recall60"}]
        )
        config = PortfolioBacktestConfig(start_date=date(2026, 1, 1), end_date=date(2026, 1, 31), strategy_name="blackbox_finetune_recall60")

        with patch.object(portfolio_db, "build_blackbox_signals", return_value=expected) as mocked:
            result = portfolio_db.load_strategy_signals(object(), config)

        mocked.assert_called_once()
        self.assertEqual(result.iloc[0]["SCode"], "000001")

    def test_stop_loss_series_builds_trade_rule_configs(self):
        args = portfolio_run.build_parser().parse_args(["--trade-rule-series", "stop_loss"])
        configs = portfolio_run.rule_configs_from_args(args)

        self.assertEqual([config.stop_loss_pct for config in configs], list(STOP_LOSS_SERIES))
        self.assertEqual(configs[1].trade_rule_name, "stop_loss_3.5pct_take_profit_10_20_hold_3d")

    def test_trade_rule_name_round_trips_stop_loss_pct(self):
        self.assertEqual(STOP_LOSS_RULE_NAMES[0], "stop_loss_3pct_take_profit_10_20_hold_3d")
        self.assertEqual(STOP_LOSS_RULE_NAMES[-1], "stop_loss_6pct_take_profit_10_20_hold_3d")
        self.assertEqual(stop_loss_rule_name(0.055), "stop_loss_5.5pct_take_profit_10_20_hold_3d")
        self.assertEqual(stop_loss_pct_from_rule_name("stop_loss_4.5pct_take_profit_10_20_hold_3d"), 0.045)

    def test_pool_run_trade_rule_parameter_sets_stop_loss(self):
        args = pool_run.build_parser().parse_args(
            [
                "--pool-source",
                "blackbox_predictions",
                "--pool-strategy-name",
                "blackbox_finetune_recall60",
                "--trade-rule",
                "stop_loss_4.5pct_take_profit_10_20_hold_3d",
            ]
        )
        config = pool_run.config_from_args(args, stop_loss_pct_from_rule_name(args.trade_rule), args.trade_rule)

        self.assertEqual(config.stop_loss_pct, 0.045)
        self.assertEqual(config.trade_rule_name, "stop_loss_4.5pct_take_profit_10_20_hold_3d")

    def test_pool_parser_accepts_blackbox_prediction_source(self):
        args = pool_run.build_parser().parse_args(
            [
                "--pool-source",
                "blackbox_predictions",
                "--pool-strategy-name",
                "blackbox_finetune_recall60",
                "--start-date",
                "20260101",
                "--end-date",
                "20260131",
            ]
        )

        self.assertEqual(args.pool_source, "blackbox_predictions")
        self.assertEqual(args.pool_strategy_name, "blackbox_finetune_recall60")

    def test_pool_parser_accepts_portfolio_backtest_source(self):
        args = pool_run.build_parser().parse_args(
            [
                "--pool-source",
                "portfolio_backtest",
                "--pool-strategy-name",
                "portfolio_precision10_0_4_1400_20260101_20260530_limit5",
                "--source-backtest-name",
                "portfolio_precision10_0_4_1400_20260101_20260530_limit5",
            ]
        )

        self.assertEqual(args.pool_source, "portfolio_backtest")
        self.assertEqual(args.source_backtest_name, "portfolio_precision10_0_4_1400_20260101_20260530_limit5")

    def test_load_portfolio_backtest_pool_uses_buy_selection_dates(self):
        class Cursor:
            def __init__(self):
                self.params = None

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def execute(self, _sql, params):
                self.params = params

            def fetchall(self):
                return [(date(2026, 1, 5), "1", "A", None, None, "source_backtest=x", "pool")]

        class Conn:
            def __init__(self):
                self.cursor_obj = Cursor()

            def cursor(self):
                return self.cursor_obj

        conn = Conn()
        frame = pool_run.load_portfolio_backtest_pool(
            conn,
            "source_bt",
            "pool",
            date(2026, 1, 1),
            date(2026, 1, 31),
            limit_per_day=None,
        )

        self.assertEqual(frame.iloc[0]["TradeDate"], date(2026, 1, 5))
        self.assertEqual(frame.iloc[0]["SCode"], "000001")
        self.assertEqual(conn.cursor_obj.params[:4], ["pool", "source_bt", date(2026, 1, 1), date(2026, 1, 31)])

    def test_pool_limit_keeps_top_scores_per_day(self):
        frame = pd.DataFrame(
            [
                {"TradeDate": "2026-01-01", "SCode": "1", "Score": 0.1},
                {"TradeDate": "2026-01-01", "SCode": "2", "Score": 0.9},
                {"TradeDate": "2026-01-01", "SCode": "3", "Score": 0.5},
                {"TradeDate": "2026-01-02", "SCode": "4", "Score": 0.2},
            ]
        )

        result = pool_run.limit_pool(frame, 2)

        self.assertEqual(result[result["TradeDate"] == date(2026, 1, 1)]["SCode"].tolist(), ["000002", "000003"])

    def test_rounds_buy_shares_to_hundred_lot(self):
        self.assertEqual(buy_shares_for_budget(10.03, 100000), 10000)
        self.assertEqual(buy_shares_for_budget(10.08, 100000), 9900)

    def test_weighted_average_price_uses_amount_volume_units(self):
        row = type("Row", (), {"Amount": 1000.0, "Volume": 10000.0, "High": 11.0, "Low": 9.0, "Close": 10.0})()
        self.assertEqual(weighted_average_price(row), 10.0)

    def test_weighted_average_price_handles_share_volume_units(self):
        row = type("Row", (), {"Amount": 10211.14, "Volume": 2439931.0, "High": 43.0, "Low": 41.0, "Close": 42.21})()

        self.assertEqual(weighted_average_price(row), 41.85)

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
        self.assertAlmostEqual(buy.Fee, buy.GrossAmount * 0.0002)
        self.assertEqual(buy.StampDuty, 0.0)
        self.assertAlmostEqual(sell.StampDuty, sell.GrossAmount * 0.0005)
        self.assertAlmostEqual(sell.Fee, sell.GrossAmount * 0.0007)
        self.assertAlmostEqual(sell.NetAmount, sell.GrossAmount - sell.Fee)
        self.assertGreater(len(holdings), 0)
        self.assertIn("TradingFee", snapshots.columns)
        self.assertIn("ActualProfit", snapshots.columns)
        self.assertAlmostEqual(
            snapshots.iloc[-1]["ActualProfit"],
            snapshots.iloc[-1]["TotalMarketValue"] - config.initial_cash,
        )

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
        self.assertEqual(sell.TradeRuleName, stop_loss_rule_name(0.03))

    def test_custom_stop_loss_rule_changes_exit_price_and_rule_name(self):
        signals = pd.DataFrame(
            [{"TradeDate": date(2026, 1, 1), "SCode": "000001", "SName": "A", "Close": 10.0, "Score": 1.0, "Reason": "x", "StrategyName": "test"}]
        )
        daily = pd.DataFrame(
            [
                ["000001", "A", date(2026, 1, 1), 10, 10, 10, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 2), 10, 10, 10, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 3), 10, 10, 12, 9.5, 1000, 10000],
            ],
            columns=["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume"],
        )
        config = PortfolioBacktestConfig(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 3),
            strategy_name="test",
            stop_loss_pct=0.05,
            trade_rule_name=stop_loss_rule_name(0.05),
        )
        snapshots, holdings, trades = simulate_portfolio(signals, daily, config, verbose=False)

        sell = trades[trades["Side"] == "SELL"].iloc[0]
        self.assertEqual(sell.Reason, "stop_loss_5pct")
        self.assertEqual(round_cent(sell.Price), 9.50)
        self.assertTrue((snapshots["TradeRuleName"] == stop_loss_rule_name(0.05)).all())
        self.assertTrue((holdings["TradeRuleName"] == stop_loss_rule_name(0.05)).all())

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

    def test_simulation_stops_after_end_date_when_flat(self):
        signals = pd.DataFrame(
            [{"TradeDate": date(2026, 1, 1), "SCode": "000001", "SName": "A", "Close": 10.0, "Score": 1.0, "Reason": "x", "StrategyName": "test"}]
        )
        daily = pd.DataFrame(
            [
                ["000001", "A", date(2026, 1, 1), 10, 10, 10, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 2), 10, 10, 10, 10, 1000, 10000],
                ["000001", "A", date(2026, 1, 3), 10, 10, 10.5, 9.8, 1000, 10000],
                ["000001", "A", date(2026, 1, 4), 10, 10, 10.5, 9.8, 1000, 10000],
                ["000001", "A", date(2026, 1, 5), 10, 10, 10.5, 9.8, 1000, 10000],
                ["000001", "A", date(2026, 1, 6), 10, 10, 10.5, 9.8, 1000, 10000],
            ],
            columns=["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume"],
        )
        config = PortfolioBacktestConfig(start_date=date(2026, 1, 1), end_date=date(2026, 1, 2), strategy_name="test")
        snapshots, _, _ = simulate_portfolio(signals, daily, config, verbose=False)

        self.assertEqual(snapshots.iloc[-1]["TradeDate"], date(2026, 1, 5))

    def test_market_value_does_not_jump_when_amount_volume_scale_differs(self):
        signals = pd.DataFrame(
            [{"TradeDate": date(2026, 1, 8), "SCode": "688350", "SName": "B", "Close": 27.0, "Score": 1.0, "Reason": "x", "StrategyName": "test"}]
        )
        daily = pd.DataFrame(
            [
                ["688350", "B", date(2026, 1, 8), 26, 27, 28, 26, 27000.0, 10000000.0],
                ["688350", "B", date(2026, 1, 9), 26, 27, 28, 26, 27000.0, 10000000.0],
            ],
            columns=["SCode", "SName", "TradeDate", "Open", "Close", "High", "Low", "Amount", "Volume"],
        )
        config = PortfolioBacktestConfig(start_date=date(2026, 1, 8), end_date=date(2026, 1, 9), strategy_name="test")
        snapshots, _, trades = simulate_portfolio(signals, daily, config, verbose=False)

        buy = trades[trades["Side"] == "BUY"].iloc[0]
        self.assertEqual(buy.Price, 27.0)
        self.assertLess(snapshots.iloc[-1]["HoldingMarketValue"], 110000)


if __name__ == "__main__":
    unittest.main()
