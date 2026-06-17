import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from blackbox_finetune_down6_neutral import build_dataset, build_validation_dataset, common, train


PROJECT_ENV_KEYS = [
    "MODEL_TAG",
    "DOWNSIDE_TAG",
    "SAMPLE_MODE",
    "TRAIN_START_DATE",
    "TRAIN_END_DATE",
    "VALIDATION_START_DATE",
    "VALIDATION_END_DATE",
    "TEST_START_DATE",
    "TEST_END_DATE",
    "NEUTRAL_RATIO",
    "NEGATIVE_RATIO",
    "DOWN6_CE_WEIGHT",
    "NEUTRAL_CE_WEIGHT",
    "DOWN6_LOW_SCORE_PENALTY",
    "DOWN6_SCORE_FLOOR",
    "DOWN6_LOW_SCORE_WEIGHT",
]


@contextmanager
def without_project_env():
    saved = {key: os.environ.pop(key, None) for key in PROJECT_ENV_KEYS}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def sample_row(scode="000001", trade_date="2026-01-05", close=10.0, highs=(10.2, 10.3, 10.4), lows=(9.5, 9.3, 9.1)):
    return (
        scode,
        trade_date,
        close,
        highs[0],
        lows[0],
        highs[1],
        lows[1],
        highs[2],
        lows[2],
    )


class BlackboxFinetuneDown6NeutralTests(unittest.TestCase):
    def test_default_periods_match_down6_task(self):
        with without_project_env():
            train_args = build_dataset.build_parser().parse_args([])
            validation_args = build_validation_dataset.build_parser().parse_args([])

        self.assertEqual(train_args.start_date, "20230101")
        self.assertEqual(train_args.end_date, "20241231")
        self.assertEqual(validation_args.start_date, "20260101")
        self.assertEqual(validation_args.end_date, "20260529")
        self.assertEqual(common.recall_target_tag(), "down6")
        self.assertIn("data_no_partial_week_down6_long", str(common.default_data_dir("long")))

    def test_classifies_three_day_down_six_as_positive_label(self):
        candidate = build_dataset._classify_candidate(sample_row(lows=(9.8, 9.5, 9.39)), symbol_index=3)

        self.assertIsNotNone(candidate)
        self.assertTrue(candidate.is_down6)
        self.assertFalse(candidate.is_positive_surge)

    def test_detects_positive_surge_for_neutral_exclusion(self):
        candidate = build_dataset._classify_candidate(sample_row(highs=(10.5, 12.1, 11.0), lows=(9.8, 9.7, 9.6)), symbol_index=7)

        self.assertIsNotNone(candidate)
        self.assertTrue(candidate.is_positive_surge)
        self.assertFalse(candidate.is_down6)

    def test_neutral_ratio_can_come_from_environment(self):
        with without_project_env(), patch.dict("os.environ", {"NEUTRAL_RATIO": "4.5"}):
            train_args = build_dataset.build_parser().parse_args([])
            validation_args = build_validation_dataset.build_parser().parse_args([])

        self.assertEqual(train_args.neutral_ratio, 4.5)
        self.assertEqual(validation_args.neutral_ratio, 4.5)

    def test_training_parser_keeps_recall60_style_defaults(self):
        args = train.build_parser().parse_args([])

        self.assertEqual(args.learning_rate, 5e-6)
        self.assertEqual(args.gradient_accumulation_steps, 16)
        self.assertEqual(args.eval_threshold, 0.48)
        self.assertEqual(args.down6_ce_weight, 3.0)
        self.assertEqual(args.neutral_ce_weight, 1.0)
        self.assertTrue(args.down6_low_score_penalty)
        self.assertEqual(args.down6_score_floor, 0.45)
        self.assertEqual(args.down6_low_score_weight, 0.2)

    def test_evaluation_reports_down6_low_score_rate(self):
        from blackbox_finetune_down6_neutral import evaluate

        summary = evaluate.summarize_scored_rows(
            [
                {"actual_label": 1, "predicted_label": 1, "positive_probability": 0.60},
                {"actual_label": 1, "predicted_label": 0, "positive_probability": 0.20},
                {"actual_label": 0, "predicted_label": 0, "positive_probability": 0.10},
            ],
            precision_top_k=2,
            down6_score_floor=0.45,
        )

        self.assertEqual(summary["positive_samples"], 2)
        self.assertEqual(summary["down6_low_score_count"], 1)
        self.assertEqual(summary["down6_low_score_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
