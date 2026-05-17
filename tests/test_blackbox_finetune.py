import json
import unittest
from datetime import date

from blackbox_finetune import build_dataset
from blackbox_finetune import common


class BlackboxFinetuneTests(unittest.TestCase):
    def test_label_answer_is_strict_json(self):
        positive = json.loads(common.label_answer(1))
        negative = json.loads(common.label_answer(0))

        self.assertEqual(positive["label"], "positive")
        self.assertEqual(negative["label"], "negative")
        self.assertGreater(positive["positive_probability"], negative["positive_probability"])

    def test_build_messages_has_supervised_answer_only_when_label_given(self):
        kline = [{"date": "20260102", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10, "amount": 20}]

        inference_messages = common.build_messages("000001", date(2026, 1, 2), kline, kline)
        train_messages = common.build_messages("000001", date(2026, 1, 2), kline, kline, 1)

        self.assertEqual([message["role"] for message in inference_messages], ["system", "user"])
        self.assertEqual([message["role"] for message in train_messages], ["system", "user", "assistant"])
        self.assertIn("daily_55", train_messages[1]["content"])
        self.assertEqual(json.loads(train_messages[-1]["content"])["label"], "positive")

    def test_split_train_test_is_stable(self):
        rows = [
            {"metadata": {"scode": f"{idx:06d}", "anchor_date": "2026-01-01", "label": idx % 2}}
            for idx in range(10)
        ]

        train_rows, test_rows = build_dataset.split_train_test(rows, 0.8, 2026)
        train_rows_2, test_rows_2 = build_dataset.split_train_test(rows, 0.8, 2026)

        self.assertEqual(len(train_rows), 8)
        self.assertEqual(len(test_rows), 2)
        self.assertEqual(train_rows, train_rows_2)
        self.assertEqual(test_rows, test_rows_2)

    def test_event_key_uses_symbol_and_date(self):
        anchor = date(2026, 1, 5)

        self.assertEqual(common.event_key("000001", anchor), ("000001", anchor))


if __name__ == "__main__":
    unittest.main()
