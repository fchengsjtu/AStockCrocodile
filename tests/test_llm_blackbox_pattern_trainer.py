import unittest
from collections import Counter
from datetime import date, timedelta

import pandas as pd

import llm_blackbox_pattern_trainer as trainer


class LlmBlackboxPatternTrainerTests(unittest.TestCase):
    def _config(self):
        return trainer.BlackboxTrainingConfig(
            stat_type="short_term_surge_3d_20pct",
            train_ratio=0.8,
            split_seed=123,
            daily_window=55,
            weekly_window=55,
            min_success_rate=0.2,
            min_sample_count=20,
            min_positive_support=5,
            min_pattern_size=3,
            max_pattern_size=8,
            candidate_count=10,
            batch_size=40,
            prompt_batch_size=20,
            max_training_batches=0,
            model="deepseek-r1-distill-qwen-14b",
            api_base_url="http://127.0.0.1:1234/v1",
            api_key_env="LOCAL_LLM_API_KEY",
            output=None,
            save_db=False,
        )

    def test_split_samples_is_stable_80_20(self):
        rows = []
        base = date(2026, 1, 1)
        for index in range(10):
            rows.append(
                {
                    "SCode": f"{index:06d}",
                    "SName": "Sample",
                    "StartRiseDate": base + timedelta(days=index),
                    "PrevTradeDate": base + timedelta(days=index - 1),
                    "SelectionDate": base + timedelta(days=index - 1),
                    "GainRate": 0.2,
                }
            )
        samples = pd.DataFrame(rows)

        train_a, test_a = trainer.split_samples(samples, 0.8, 123)
        train_b, test_b = trainer.split_samples(samples, 0.8, 123)

        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(test_a), 2)
        self.assertEqual(train_a["SCode"].tolist(), train_b["SCode"].tolist())
        self.assertEqual(test_a["SCode"].tolist(), test_b["SCode"].tolist())

    def test_blackbox_prompt_uses_prev_trade_date_anchor(self):
        prompt = trainer.build_blackbox_prompt(
            [
                {
                    "scode": "000001",
                    "anchor_date": "2026-05-11",
                    "daily_55": {"rows": [[0, 100, -100, 0, 120]]},
                    "weekly_55": {"rows": [[0, 100, -100, 0, 90]]},
                }
            ],
            Counter({("D_CLOSE_GT_MA5",): 3, ("W_MA5_GT_MA13",): 2, ("D_RET_5_GE_5",): 2}),
            self._config(),
            batch_no=1,
        )

        self.assertIn("PrevTradeDate", prompt)
        self.assertIn('"anchor_date": "2026-05-11"', prompt)
        self.assertIn("D_CLOSE_GT_MA5", prompt)
        self.assertIn('"minimum_required_validated_success_rate": 0.2', prompt)

    def test_compact_window_keeps_ohlcv_rows_small(self):
        base = date(2026, 1, 1)
        frame = pd.DataFrame(
            [
                {
                    "TradeDate": base + timedelta(days=index),
                    "Open": 10 + index,
                    "High": 11 + index,
                    "Low": 9 + index,
                    "Close": 10 + index,
                    "Volume": 1000 + index,
                }
                for index in range(55)
            ]
        )

        compact = trainer._compact_window_to_matrix(frame)

        self.assertEqual(len(compact["rows"]), 55)
        self.assertEqual(compact["columns"], ["open_bp", "high_bp", "low_bp", "close_bp", "volume_pct_avg"])
        self.assertLess(len(str(compact)), 3000)

    def test_events_as_anchor_labels_uses_prev_trade_date(self):
        events = pd.DataFrame(
            [
                {
                    "SCode": "000001",
                    "PrevTradeDate": date(2026, 5, 11),
                    "SelectionDate": date(2026, 5, 8),
                }
            ]
        )

        labels = trainer._events_as_anchor_labels(events)

        self.assertEqual(labels.iloc[0]["SelectionDate"], date(2026, 5, 11))


if __name__ == "__main__":
    unittest.main()
