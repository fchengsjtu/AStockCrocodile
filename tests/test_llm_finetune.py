import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from llm_finetune import build_dataset
from llm_finetune import common
from llm_finetune import evaluate
from llm_finetune import train


class LlmFinetuneTests(unittest.TestCase):
    def test_split_80_20_is_stable(self):
        rows = [
            {"metadata": {"scode": f"{idx:06d}", "anchor_date": "2026-01-01", "label": idx % 2}}
            for idx in range(10)
        ]

        train_rows, test_rows = build_dataset.split_80_20(rows, 123)
        train_rows_2, test_rows_2 = build_dataset.split_80_20(rows, 123)

        self.assertEqual(len(train_rows), 8)
        self.assertEqual(len(test_rows), 2)
        self.assertEqual(train_rows, train_rows_2)
        self.assertEqual(test_rows, test_rows_2)

    def test_pick_window_uses_anchor_and_exact_window(self):
        rows = [{"date": (date(2026, 1, 1) + timedelta(days=idx)).strftime("%Y%m%d")} for idx in range(60)]

        window = common.pick_window(rows, date(2026, 3, 1), 55)

        self.assertEqual(len(window), 55)
        self.assertEqual(window[-1]["date"], "20260301")

    def test_messages_include_supervised_json_label(self):
        kline = [{"date": "20260101", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10, "amount": 20}]

        messages = common.build_messages("000001", date(2026, 1, 1), kline, kline, 1)
        answer = json.loads(messages[-1]["content"])

        self.assertEqual(answer["label"], "positive")
        self.assertIn("daily_55", messages[1]["content"])

    def test_extract_json(self):
        parsed = evaluate.extract_json("prefix {\"label\":\"positive\",\"success_probability\":0.5} suffix")

        self.assertEqual(parsed["label"], "positive")
        self.assertEqual(parsed["success_probability"], 0.5)

    def test_missing_adapter_error_mentions_train_command(self):
        message = str(evaluate.missing_adapter_error(Path("llm_finetune/runs/test/adapter")))

        self.assertIn("adapter_config.json", message)
        self.assertIn("llm_finetune.train", message)

    def test_normalize_training_args_supports_eval_strategy_rename(self):
        kwargs = train.normalize_training_args({"evaluation_strategy": "epoch", "output_dir": "x"}, {"eval_strategy", "output_dir"})

        self.assertEqual(kwargs["eval_strategy"], "epoch")
        self.assertNotIn("evaluation_strategy", kwargs)

    def test_write_and_read_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            common.write_jsonl(path, [{"a": 1}, {"b": 2}])

            rows = common.read_jsonl(path)

        self.assertEqual(rows, [{"a": 1}, {"b": 2}])


if __name__ == "__main__":
    unittest.main()
