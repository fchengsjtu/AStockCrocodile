import unittest
from datetime import date, timedelta

from blackbox_finetune_recall80 import build_dataset, build_validation_dataset, common


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
    def test_training_defaults_match_partial_week_task(self):
        args = build_dataset.build_parser().parse_args([])

        self.assertEqual(args.start_date, "20110101")
        self.assertEqual(args.end_date, "20241231")
        self.assertEqual(str(args.output_dir), "blackbox_finetune_recall80\\data_partial_week")

    def test_validation_defaults_match_holdout_period(self):
        args = build_validation_dataset.build_parser().parse_args([])

        self.assertEqual(args.start_date, "20260101")
        self.assertEqual(args.end_date, "20260430")
        self.assertEqual(str(args.output_dir), "blackbox_finetune_recall80\\data_validation_partial_week")

    def test_pick_weekly_window_appends_temporary_week_for_midweek_anchor(self):
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
        self.assertEqual(picked[-1]["date"], "20260408")
        self.assertEqual(picked[-1]["open"], 20)
        self.assertEqual(picked[-1]["high"], 25)
        self.assertEqual(picked[-1]["low"], 19)
        self.assertEqual(picked[-1]["close"], 24)
        self.assertEqual(picked[-1]["volume"], 350)
        self.assertEqual(picked[-1]["amount"], 3500)


if __name__ == "__main__":
    unittest.main()
