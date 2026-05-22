import unittest
from datetime import date

import pandas as pd

from blackbox_finetune.prediction_store import save_top_predictions
from portfolio_backtest import track_blackbox


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))

    def executemany(self, sql, rows):
        self.conn.executemany_calls.append((sql, list(rows)))


class FakeConnection:
    def __init__(self):
        self.executed = []
        self.executemany_calls = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


class BlackboxPredictionStoreTests(unittest.TestCase):
    def test_save_top_predictions_keeps_top_five_and_strategy(self):
        conn = FakeConnection()
        predictions = pd.DataFrame(
            [
                {"TradeDate": date(2026, 5, 14), "SCode": f"00000{i}", "PositiveProbability": prob, "PositiveLoss": 1.0, "NegativeLoss": 2.0}
                for i, prob in enumerate([0.1, 0.9, 0.8, 0.7, 0.6, 0.5], start=1)
            ]
        )

        count = save_top_predictions(conn, predictions, "blackbox_finetune_recall30", 0.5, 512, top_n=5)

        self.assertEqual(count, 5)
        rows = conn.executemany_calls[-1][1]
        self.assertEqual([row[2] for row in rows], [1, 2, 3, 4, 5])
        self.assertEqual(rows[0][1], "blackbox_finetune_recall30")
        self.assertEqual(rows[0][3], "000002")
        self.assertTrue(any("DELETE FROM blackbox_predictions" in sql for sql, _ in conn.executed))

    def test_track_blackbox_parser_requires_strategy(self):
        args = track_blackbox.build_parser().parse_args(["--strategy-name", "blackbox_finetune_recall30"])

        self.assertEqual(args.strategy_name, "blackbox_finetune_recall30")
        self.assertEqual(args.initial_cash, 1_000_000.0)


if __name__ == "__main__":
    unittest.main()
