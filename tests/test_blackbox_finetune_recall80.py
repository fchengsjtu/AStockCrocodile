import unittest
from datetime import date, timedelta

from blackbox_finetune_recall80 import build_dataset, build_validation_dataset, common, evaluate, predict_day, train


def daily(day, open_, high, low, close, volume=100.0, amount=1000.0):
    return {
        "date": day,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
        "ma5": close,
        "ma13": close,
        "ma34": close,
        "ma55": close,
    }


def weekly(day, close):
    return daily(day, close, close, close, close)


class BlackboxFinetuneRecall80Tests(unittest.TestCase):
    def test_training_defaults_match_no_partial_week_task(self):
        args = build_dataset.build_parser().parse_args([])

        self.assertEqual(args.start_date, "20110101")
        self.assertEqual(args.end_date, "20241231")
        self.assertEqual(args.output_dir, common.default_data_dir("long"))

    def test_validation_defaults_match_holdout_period(self):
        args = build_validation_dataset.build_parser().parse_args([])

        self.assertEqual(args.start_date, "20260101")
        self.assertEqual(args.end_date, "20260430")
        self.assertEqual(args.output_dir, common.default_validation_dir("long"))

    def test_tokenization_uses_compact_csv_scheme_with_sample_modes(self):
        self.assertEqual(common.COMPACT_DAILY_WINDOW, 13)
        self.assertEqual(common.COMPACT_WEEKLY_WINDOW, 8)
        self.assertEqual(common.COMPACT_MONTHLY_WINDOW, 5)
        self.assertEqual(common.default_max_seq_length("short"), 1024)
        self.assertEqual(train.build_parser().parse_args([]).max_seq_length, 2048)
        self.assertEqual(evaluate.build_parser().parse_args([]).max_seq_length, 2048)
        self.assertEqual(predict_day.build_parser().parse_args(["--date", "20260514"]).max_seq_length, 2048)

    def test_compact_prompt_normalizes_prices_volume_and_amount(self):
        daily_rows = [
            daily("20260101", 1, 2, 1, 2, volume=100, amount=1000),
            daily("20260102", 2, 3, 1, 2, volume=200, amount=3000),
        ]
        weekly_rows = [
            daily("20260102", 1, 2, 1, 2, volume=10, amount=50),
            daily("20260109", 2, 3, 1, 2, volume=30, amount=150),
        ]

        prompt = common.build_compact_prompt("000001", date(2026, 1, 10), daily_rows, weekly_rows, daily_window=2, weekly_window=2, monthly_window=0)

        self.assertIn("cols=dt/o/h/l/c/v/a/m5/m13/m34/m55", prompt)
        self.assertIn("D\n1,0.5,1,0.5,1,0.67,0.5", prompt)
        self.assertIn("2,1,1.5,0.5,1,1.33,1.5", prompt)
        self.assertIn("W\n1,0.5,1,0.5,1,0.5,0.5", prompt)
        self.assertIn("2,1,1.5,0.5,1,1.5,1.5", prompt)

    def test_pick_weekly_window_uses_only_completed_weekly_rows_for_midweek_anchor(self):
        first_week = date(2025, 3, 21)
        weekly_rows = [weekly((first_week + timedelta(days=7 * i)).strftime("%Y%m%d"), 10 + i) for i in range(55)]
        daily_rows = [
            daily("20260406", 20, 22, 19, 21, 100, 1000),
            daily("20260407", 21, 24, 20, 23, 120, 1200),
            daily("20260408", 23, 25, 22, 24, 130, 1300),
        ]

        picked = common.pick_weekly_window(weekly_rows, daily_rows, date(2026, 4, 8), 55)

        self.assertIsNotNone(picked)
        self.assertEqual(len(picked), 55)
        self.assertEqual(picked[-1]["date"], "20260403")
        self.assertNotEqual(picked[-1]["date"], "20260408")


if __name__ == "__main__":
    unittest.main()
