import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from llm_finetune import build_dataset
from llm_finetune import evaluate_model
from llm_finetune.common import KlineWindowSample, build_response, choose_target_features, to_messages_jsonl, write_jsonl


class LlmFineTuneTests(unittest.TestCase):
    def test_choose_target_features_prefers_returns_and_volume(self):
        features = choose_target_features(
            {
                "W_MA5_GT_MA13",
                "D_CLOSE_GT_MA5",
                "D_RET_5_GE_5",
                "W_VOL_RATIO_GE_1_5",
                "D_RANGE_10_CONTRACT",
            },
            min_size=3,
            max_size=4,
        )

        self.assertEqual(len(features), 4)
        self.assertIn("D_RET_5_GE_5", features)
        self.assertIn("W_VOL_RATIO_GE_1_5", features)

    def test_build_response_uses_negative_empty_patterns(self):
        sample = KlineWindowSample(
            scode="000001",
            trade_date=date(2026, 1, 2),
            label="negative",
            gain_rate=None,
            features=("D_CLOSE_GT_MA5", "D_RET_5_GE_5", "W_MA5_GT_MA13"),
            daily_55={"rows": []},
            weekly_55={"rows": []},
        )

        payload = json.loads(build_response(sample))

        self.assertEqual(payload, {"label": "negative", "patterns": []})

    def test_messages_jsonl_contains_chat_messages(self):
        sample = KlineWindowSample(
            scode="000001",
            trade_date=date(2026, 1, 2),
            label="positive",
            gain_rate=0.22,
            features=("D_CLOSE_GT_MA5", "D_RET_5_GE_5", "W_MA5_GT_MA13"),
            daily_55={"rows": [[0, 1, -1, 0, 100]]},
            weekly_55={"rows": [[0, 1, -1, 0, 100]]},
        )

        row = to_messages_jsonl(sample, sample.features)

        self.assertEqual([item["role"] for item in row["messages"]], ["system", "user", "assistant"])
        self.assertEqual(row["metadata"]["label"], "positive")
        self.assertIn("allowed_feature_tokens", row["messages"][1]["content"])

    def test_write_jsonl_writes_one_json_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            count = write_jsonl(path, [{"a": 1}, {"b": 2}])

            self.assertEqual(count, 2)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_split_rows_is_stable(self):
        rows = [
            KlineWindowSample(str(index), date(2026, 1, index + 1), "positive", 0.2, (), {}, {})
            for index in range(10)
        ]

        train_a, valid_a = build_dataset.split_rows(rows, 0.2, 123)
        train_b, valid_b = build_dataset.split_rows(rows, 0.2, 123)

        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(valid_a), 2)
        self.assertEqual([row.scode for row in valid_a], [row.scode for row in valid_b])

    def test_evaluate_default_min_success_rate_is_40_percent(self):
        self.assertEqual(evaluate_model.DEFAULT_MIN_SUCCESS_RATE, 0.40)


if __name__ == "__main__":
    unittest.main()
