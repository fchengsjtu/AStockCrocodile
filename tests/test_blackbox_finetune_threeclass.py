from __future__ import annotations

import unittest

from blackbox_finetune_threeclass.build_dataset import (
    FutureBar,
    classify_future_path,
    rebalance_materialized_samples,
)
from blackbox_finetune_threeclass.common import (
    CLASS_NEGATIVE,
    CLASS_NEUTRAL,
    CLASS_POSITIVE,
    compact_messages_from_sample,
    label_answer,
)
from blackbox_finetune_threeclass.build_dataset import build_parser as build_dataset_parser
from blackbox_finetune_threeclass.inference import probabilities_from_losses
from blackbox_finetune_threeclass.metrics import summarize_scored_rows


def sample(label: int, index: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "old"},
            {"role": "user", "content": f"s=000001\nD\n{index}"},
            {"role": "assistant", "content": "old"},
        ],
        "metadata": {
            "scode": f"{index:06d}",
            "anchor_date": f"2026-01-{(index % 28) + 1:02d}",
            "label": label,
        },
    }


class ThreeClassTests(unittest.TestCase):
    def test_defaults_follow_current_recall60_xlong_run(self):
        args = build_dataset_parser().parse_args([])
        self.assertEqual(args.start_date, "20230101")
        self.assertEqual(args.end_date, "20241231")
        self.assertEqual(args.sample_mode, "xlong")

    def test_future_path_uses_first_trigger(self):
        self.assertEqual(
            classify_future_path(10.0, [FutureBar(10.5, 9.7), FutureBar(12.1, 9.5), FutureBar(10.0, 9.0)]),
            CLASS_POSITIVE,
        )
        self.assertEqual(
            classify_future_path(10.0, [FutureBar(10.5, 9.3), FutureBar(12.5, 9.8), FutureBar(10.0, 9.8)]),
            CLASS_NEGATIVE,
        )
        self.assertEqual(
            classify_future_path(10.0, [FutureBar(10.5, 9.5), FutureBar(11.0, 9.6), FutureBar(11.9, 9.5)]),
            CLASS_NEUTRAL,
        )

    def test_same_bar_dual_trigger_is_discarded(self):
        self.assertIsNone(
            classify_future_path(10.0, [FutureBar(12.1, 9.3), FutureBar(10.0, 9.8), FutureBar(10.0, 9.8)])
        )

    def test_rebalance_is_exactly_one_two_ten(self):
        rows = (
            [sample(CLASS_POSITIVE, index) for index in range(4)]
            + [sample(CLASS_NEGATIVE, index + 100) for index in range(20)]
            + [sample(CLASS_NEUTRAL, index + 200) for index in range(80)]
        )
        selected = rebalance_materialized_samples(rows, seed=7)
        counts = {label: sum(int(row["metadata"]["label"]) == label for row in selected) for label in range(3)}
        self.assertEqual(counts, {CLASS_NEGATIVE: 8, CLASS_NEUTRAL: 40, CLASS_POSITIVE: 4})

    def test_three_class_answers_are_compact(self):
        self.assertEqual(label_answer(CLASS_POSITIVE), '{"c":"positive"}')
        self.assertEqual(label_answer(CLASS_NEGATIVE), '{"c":"negative"}')
        self.assertEqual(label_answer(CLASS_NEUTRAL), '{"c":"neutral"}')
        messages = compact_messages_from_sample(sample(CLASS_NEUTRAL, 1))
        self.assertEqual(messages[-1]["content"], '{"c":"neutral"}')

    def test_probabilities_are_normalized_and_follow_loss(self):
        probabilities = probabilities_from_losses({CLASS_NEGATIVE: 2.0, CLASS_NEUTRAL: 1.0, CLASS_POSITIVE: 0.5})
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities[CLASS_POSITIVE], probabilities[CLASS_NEUTRAL])
        self.assertGreater(probabilities[CLASS_NEUTRAL], probabilities[CLASS_NEGATIVE])

    def test_metrics_include_confusion_and_positive_precision(self):
        rows = [
            {"actual_label": CLASS_POSITIVE, "predicted_label": CLASS_POSITIVE, "positive_probability": 0.9},
            {"actual_label": CLASS_NEGATIVE, "predicted_label": CLASS_POSITIVE, "positive_probability": 0.8},
            {"actual_label": CLASS_NEUTRAL, "predicted_label": CLASS_NEUTRAL, "positive_probability": 0.1},
        ]
        summary = summarize_scored_rows(rows, (2,))
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["confusion_matrix"]["positive"]["positive"], 1)
        self.assertAlmostEqual(summary["positive_precision@2"], 0.5)


if __name__ == "__main__":
    unittest.main()
