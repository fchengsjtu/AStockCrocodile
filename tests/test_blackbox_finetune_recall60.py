import unittest
from datetime import date
from unittest.mock import patch

from blackbox_finetune import build_dataset as base_build_dataset
from blackbox_finetune.common import SampleEvent
from blackbox_finetune_recall60 import build_dataset, build_validation_dataset, common, evaluate, predict_day, train


def daily(day, open_, high, low, close, volume=100.0, amount=1000.0):
    return {
        "date": day,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
        "ma5": 0.0,
        "ma13": 0.0,
        "ma34": 0.0,
        "ma55": 0.0,
    }


class BlackboxFinetuneRecall60Tests(unittest.TestCase):
    def test_training_defaults_use_2020_to_2025(self):
        args = build_dataset.build_parser().parse_args([])

        self.assertEqual(args.start_date, "20200101")
        self.assertEqual(args.end_date, "20251231")
        self.assertEqual(args.output_dir, common.DEFAULT_DATA_DIR)
        self.assertEqual(args.negative_ratio, 3.0)

    def test_negative_ratio_default_can_come_from_environment(self):
        with patch.dict("os.environ", {"NEGATIVE_RATIO": "2.5"}):
            build_args = build_dataset.build_parser().parse_args([])
            validation_args = build_validation_dataset.build_parser().parse_args([])

        self.assertEqual(build_args.negative_ratio, 2.5)
        self.assertEqual(validation_args.negative_ratio, 2.5)

    def test_positive_samples_use_twenty_trading_day_cooldown(self):
        trade_dates = [date(2026, 1, day) for day in range(1, 32)]
        events = [
            SampleEvent("000001", date(2026, 1, 1), 1, "positive", 1.0),
            SampleEvent("000001", date(2026, 1, 10), 1, "positive", 2.0),
            SampleEvent("000001", date(2026, 1, 22), 1, "positive", 3.0),
            SampleEvent("000002", date(2026, 1, 10), 1, "positive", 4.0),
        ]

        with patch.object(
            base_build_dataset,
            "load_trading_dates_for_symbols_batched",
            return_value={"000001": trade_dates, "000002": trade_dates},
        ):
            kept = base_build_dataset.apply_positive_cooldown(None, events, 20)

        self.assertEqual([(event.scode, event.anchor_date) for event in kept], [
            ("000001", date(2026, 1, 1)),
            ("000001", date(2026, 1, 22)),
            ("000002", date(2026, 1, 10)),
        ])
        self.assertEqual(base_build_dataset.NEGATIVE_EXCLUSION_TRADING_DAYS, 20)

    def test_evaluation_defaults_match_2026_holdout_period(self):
        args = build_validation_dataset.build_parser().parse_args([])

        self.assertEqual(args.start_date, "20260101")
        self.assertEqual(args.end_date, "20260430")
        self.assertEqual(args.output_dir, common.DEFAULT_VALIDATION_DIR)
        self.assertIn("no_partial", str(args.output_dir))

    def test_compact_window_and_sequence_length_defaults_match_2048_format(self):
        self.assertEqual(common.COMPACT_DAILY_WINDOW, 13)
        self.assertEqual(common.COMPACT_WEEKLY_WINDOW, 8)
        self.assertEqual(common.COMPACT_MONTHLY_WINDOW, 5)
        self.assertEqual(common.sample_mode_config("short")["daily"], 8)
        self.assertEqual(common.sample_mode_config("short")["weekly"], 5)
        self.assertEqual(common.default_max_seq_length("short"), 1024)
        self.assertEqual(common.sample_mode_config("xlong"), {"daily": 21, "weekly": 13, "monthly": 8, "max_seq_length": 3072})
        self.assertEqual(common.default_max_seq_length("xlong"), 3072)
        self.assertEqual(common.sample_mode_config("xxlong"), {"daily": 34, "weekly": 21, "monthly": 13, "max_seq_length": 4096})
        self.assertEqual(common.default_max_seq_length("xxlong"), 4096)
        self.assertEqual(train.build_parser().parse_args([]).max_seq_length, 2048)
        self.assertEqual(evaluate.build_parser().parse_args([]).max_seq_length, 2048)
        self.assertEqual(predict_day.build_parser().parse_args(["--date", "20260514"]).max_seq_length, 2048)

    def test_pick_weekly_window_uses_only_completed_weekly_rows_for_midweek_anchor(self):
        weekly_rows = [
            daily("20251205", 1, 1, 1, 10),
            daily("20251212", 1, 1, 1, 11),
            daily("20251219", 1, 1, 1, 12),
            daily("20251226", 1, 1, 1, 13),
            daily("20260102", 1, 1, 1, 14),
        ]
        daily_rows = [
            daily("20251229", 14, 15, 13, 14),
            daily("20251230", 14, 16, 13, 15),
            daily("20251231", 15, 17, 14, 16),
        ]

        window = common.pick_weekly_window(weekly_rows, daily_rows, date(2025, 12, 31), 4)

        self.assertIsNotNone(window)
        self.assertEqual([row["date"] for row in window], ["20251205", "20251212", "20251219", "20251226"])
        self.assertEqual(window[-1]["close"], 13)

    def test_compact_prompt_uses_short_csv_recent_thirteen_daily_eight_weekly_and_five_monthly(self):
        daily_rows = [daily(f"202601{day:02d}", day, day + 1, day - 1, day + 0.5) for day in range(1, 26)]
        weekly_rows = [daily(f"2025{week:02d}01", week, week + 1, week - 1, week + 0.5) for week in range(1, 16)]
        monthly_rows = [daily(f"2025{month:02d}28", month, month + 1, month - 1, month + 0.5) for month in range(1, 8)]

        prompt = common.build_compact_prompt("000001", date(2026, 1, 10), daily_rows, weekly_rows, monthly_rows)

        self.assertIn("cols=dt/o/h/l/c/v/a/m5/m13/m34/m55", prompt)
        self.assertEqual(prompt.splitlines().count("D"), 1)
        self.assertEqual(prompt.splitlines().count("W"), 1)
        self.assertEqual(prompt.splitlines().count("M"), 1)
        daily_section = prompt.split("D\n", 1)[1].split("\nW\n", 1)[0].splitlines()
        weekly_section = prompt.split("\nW\n", 1)[1].split("\nM\n", 1)[0].splitlines()
        monthly_section = prompt.split("\nM\n", 1)[1].splitlines()
        self.assertEqual(len(daily_section), 13)
        self.assertEqual(len(weekly_section), 8)
        self.assertEqual(len(monthly_section), 5)
        self.assertTrue(daily_section[0].startswith("1,"))
        self.assertTrue(daily_section[-1].startswith("13,"))
        self.assertTrue(weekly_section[0].startswith("1,"))
        self.assertTrue(weekly_section[-1].startswith("8,"))
        self.assertTrue(monthly_section[-1].startswith("5,"))
        self.assertNotIn("260101", prompt)
        self.assertNotIn("260102", prompt)
        self.assertNotIn("260103", prompt)
        self.assertNotIn("260104", prompt)
        self.assertNotIn("260110", prompt)
        self.assertNotIn("250101", prompt)
        self.assertNotIn("250201", prompt)
        self.assertNotIn("250301", prompt)
        self.assertNotIn("250901", prompt)

    def test_compact_prompt_normalizes_prices_volume_and_amount_by_window_average(self):
        daily_rows = [
            daily("20260101", 1, 2, 1, 2, volume=100, amount=1000),
            daily("20260102", 2, 3, 1, 2, volume=200, amount=3000),
        ]
        weekly_rows = [
            daily("20260102", 1, 2, 1, 2, volume=10, amount=50),
            daily("20260109", 2, 3, 1, 2, volume=30, amount=150),
        ]

        prompt = common.build_compact_prompt("000001", date(2026, 1, 10), daily_rows, weekly_rows, daily_window=2, weekly_window=2, monthly_window=0)

        self.assertIn("D\n1,0.5,1,0.5,1,0.67,0.5", prompt)
        self.assertIn("2,1,1.5,0.5,1,1.33,1.5", prompt)
        self.assertIn("W\n1,0.5,1,0.5,1,0.5,0.5", prompt)
        self.assertIn("2,1,1.5,0.5,1,1.5,1.5", prompt)

    def test_sample_modes_require_weekly_ma13_or_monthly_rows(self):
        weekly_without_ma13 = [daily(f"202512{day:02d}", 1, 1, 1, 1) for day in range(1, 6)]
        weekly_with_ma13 = [dict(row, ma13=1.0) for row in weekly_without_ma13]
        monthly_rows = [daily(f"2025{month:02d}28", 1, 1, 1, 1) for month in range(1, 6)]

        self.assertFalse(common._sample_windows_are_valid("short", weekly_without_ma13, []))
        self.assertTrue(common._sample_windows_are_valid("short", weekly_with_ma13, []))
        self.assertFalse(common._sample_windows_are_valid("long", weekly_with_ma13, monthly_rows[:4]))
        self.assertTrue(common._sample_windows_are_valid("long", weekly_with_ma13, monthly_rows))


if __name__ == "__main__":
    unittest.main()
