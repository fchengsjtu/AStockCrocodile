import unittest
from datetime import date

from blackbox_finetune_recall60 import build_dataset, build_validation_dataset, common


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
    def test_training_defaults_end_at_2024(self):
        args = build_dataset.build_parser().parse_args([])

        self.assertEqual(args.start_date, "20110101")
        self.assertEqual(args.end_date, "20241231")
        self.assertEqual(args.output_dir, common.DEFAULT_DATA_DIR)

    def test_validation_defaults_match_holdout_period(self):
        args = build_validation_dataset.build_parser().parse_args([])

        self.assertEqual(args.start_date, "20260101")
        self.assertEqual(args.end_date, "20260430")
        self.assertEqual(args.output_dir, common.DEFAULT_VALIDATION_DIR)

    def test_build_partial_weekly_bar_for_monday_to_anchor(self):
        rows = [
            daily("20260105", 10.0, 11.0, 9.5, 10.5, 100.0, 1000.0),
            daily("20260106", 10.5, 12.0, 10.2, 11.5, 200.0, 2200.0),
            daily("20260107", 11.5, 11.8, 10.8, 11.0, 150.0, 1650.0),
        ]

        bar = common.build_partial_weekly_bar(rows, date(2026, 1, 7))

        self.assertEqual(bar["date"], "20260107")
        self.assertEqual(bar["open"], 10.0)
        self.assertEqual(bar["high"], 12.0)
        self.assertEqual(bar["low"], 9.5)
        self.assertEqual(bar["close"], 11.0)
        self.assertEqual(bar["volume"], 450.0)
        self.assertEqual(bar["amount"], 4850.0)

    def test_no_partial_weekly_bar_for_friday_anchor(self):
        rows = [daily("20260109", 10.0, 11.0, 9.5, 10.5)]

        self.assertIsNone(common.build_partial_weekly_bar(rows, date(2026, 1, 9)))

    def test_pick_weekly_window_appends_temporary_week_for_midweek_anchor(self):
        weekly_rows = [
            daily("20251205", 1, 1, 1, 10),
            daily("20251212", 1, 1, 1, 11),
            daily("20251219", 1, 1, 1, 12),
            daily("20251226", 1, 1, 1, 13),
        ]
        daily_rows = [
            daily("20251229", 14, 15, 13, 14),
            daily("20251230", 14, 16, 13, 15),
            daily("20251231", 15, 17, 14, 16),
        ]

        window = common.pick_weekly_window(weekly_rows, daily_rows, date(2025, 12, 31), 5)

        self.assertIsNotNone(window)
        self.assertEqual(window[-1]["date"], "20251231")
        self.assertEqual(window[-1]["open"], 14)
        self.assertEqual(window[-1]["high"], 17)
        self.assertEqual(window[-1]["low"], 13)
        self.assertEqual(window[-1]["close"], 16)
        self.assertEqual(window[-1]["ma5"], (10 + 11 + 12 + 13 + 16) / 5)

    def test_compact_prompt_uses_short_csv_recent_seven_daily_and_weekly(self):
        daily_rows = [daily(f"202601{day:02d}", day, day + 1, day - 1, day + 0.5) for day in range(1, 11)]
        weekly_rows = [daily(f"20250{month}01", month, month + 1, month - 1, month + 0.5) for month in range(1, 10)]

        prompt = common.build_compact_prompt("000001", date(2026, 1, 10), daily_rows, weekly_rows)

        self.assertIn("cols=dt/o/h/l/c/v/a/m5/m13/m34/m55", prompt)
        self.assertIn("D\n1,4,", prompt)
        self.assertIn("7,10,", prompt)
        self.assertIn("W\n1,3,", prompt)
        self.assertIn("7,9,", prompt)
        self.assertNotIn("260101", prompt)
        self.assertNotIn("260102", prompt)
        self.assertNotIn("260103", prompt)
        self.assertNotIn("260104", prompt)
        self.assertNotIn("260110", prompt)
        self.assertNotIn("250101", prompt)
        self.assertNotIn("250201", prompt)
        self.assertNotIn("250301", prompt)
        self.assertNotIn("250901", prompt)


if __name__ == "__main__":
    unittest.main()
