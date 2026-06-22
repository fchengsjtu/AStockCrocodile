from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from blackbox_finetune_threeclass.build_dataset import (
    FutureBar,
    _candidate_sql,
    _load_candidate_rows,
    classify_future_path,
    interleave_class_rows,
    rebalance_materialized_samples,
)
from blackbox_finetune_threeclass.common import (
    CLASS_NEGATIVE,
    CLASS_NEUTRAL,
    CLASS_POSITIVE,
    compact_messages_from_sample,
    label_answer,
    parse_date,
)
from blackbox_finetune_threeclass.build_dataset import build_parser as build_dataset_parser
from blackbox_finetune_threeclass.build_validation_dataset import main as build_validation_main
from blackbox_finetune_threeclass.inference import probabilities_from_losses, selection_score
from blackbox_finetune_threeclass.predict_day import (
    build_parser as build_predict_parser,
    rank_predictions,
)
from blackbox_finetune_threeclass.metrics import (
    positive_probability_top_rows,
    selection_score_top_rows,
    summarize_scored_rows,
)
from blackbox_finetune_threeclass import train as threeclass_train


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


def sample_on_date(label: int, index: int, anchor_date: str) -> dict:
    row = sample(label, index)
    row["metadata"]["anchor_date"] = anchor_date
    return row


class ThreeClassTests(unittest.TestCase):
    def test_defaults_follow_current_recall60_xlong_run(self):
        args = build_dataset_parser().parse_args([])
        self.assertEqual(args.start_date, "20230101")
        self.assertEqual(args.end_date, "20241231")
        self.assertEqual(args.sample_mode, "xlong")
        self.assertEqual(args.candidate_batch_size, 80)
        self.assertEqual(args.mysql_query_retries, 3)

    def test_candidate_sql_limits_window_query_to_symbol_batch(self):
        sql = _candidate_sql(3)
        self.assertIn("SCode IN (%s,%s,%s)", sql)
        self.assertIn("LEAD(High, 3)", sql)

    def test_candidate_query_reconnects_and_retries_lost_connection(self):
        expected = [("000001", "2026-01-05", 10, 11, 9, 12, 9, 13, 9)]

        class Cursor:
            def __init__(self, connection):
                self.connection = connection

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def execute(self, sql, params):
                self.connection.attempts += 1
                if self.connection.attempts == 1:
                    raise RuntimeError(2013, "lost connection")

            def fetchall(self):
                return expected

        class Connection:
            attempts = 0
            ping_count = 0

            def cursor(self):
                return Cursor(self)

            def ping(self, reconnect=False):
                self.ping_count += int(reconnect)

        connection = Connection()
        with patch("blackbox_finetune_threeclass.build_dataset.time.sleep"):
            rows = _load_candidate_rows(
                connection,
                ["000001"],
                parse_date("20260101"),
                parse_date("20260131"),
                parse_date("20260101"),
                parse_date("20260120"),
                3,
            )
        self.assertEqual(rows, expected)
        self.assertEqual(connection.attempts, 2)
        self.assertEqual(connection.ping_count, 1)

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

    def test_rebalance_is_exactly_one_four_eleven(self):
        rows = (
            [sample_on_date(CLASS_POSITIVE, index, "2026-01-05") for index in range(4)]
            + [sample_on_date(CLASS_NEGATIVE, index + 100, "2026-01-05") for index in range(20)]
            + [sample_on_date(CLASS_NEUTRAL, index + 200, "2026-01-05") for index in range(80)]
        )
        selected = rebalance_materialized_samples(rows, seed=7)
        counts = {label: sum(int(row["metadata"]["label"]) == label for row in selected) for label in range(3)}
        self.assertEqual(counts, {CLASS_NEGATIVE: 16, CLASS_NEUTRAL: 44, CLASS_POSITIVE: 4})
        for start in range(0, len(selected), 16):
            self.assertEqual({row["metadata"]["anchor_date"] for row in selected[start : start + 16]}, {"2026-01-05"})

    def test_rebalance_requires_same_day_negative_and_neutral_rows(self):
        rows = (
            [sample_on_date(CLASS_POSITIVE, index, "2026-01-05") for index in range(2)]
            + [sample_on_date(CLASS_NEGATIVE, index + 100, "2026-01-06") for index in range(8)]
            + [sample_on_date(CLASS_NEUTRAL, index + 200, "2026-01-06") for index in range(22)]
        )
        with self.assertRaisesRegex(RuntimeError, "strict same-date"):
            rebalance_materialized_samples(rows, seed=7)

    def test_interleave_class_rows_uses_one_four_eleven_pattern(self):
        rows = (
            [sample_on_date(CLASS_POSITIVE, index, "2026-01-05") for index in range(2)]
            + [sample_on_date(CLASS_NEGATIVE, index + 100, "2026-01-05") for index in range(8)]
            + [sample_on_date(CLASS_NEUTRAL, index + 200, "2026-01-05") for index in range(22)]
        )
        ordered = interleave_class_rows(rows, 11, "test")
        labels = [int(row["metadata"]["label"]) for row in ordered[:16]]
        self.assertEqual(labels, [CLASS_POSITIVE] + [CLASS_NEGATIVE] * 4 + [CLASS_NEUTRAL] * 11)
        self.assertEqual({row["metadata"]["anchor_date"] for row in ordered[:16]}, {"2026-01-05"})

    def test_interleave_class_rows_drops_incomplete_same_day_cycles(self):
        rows = (
            [sample_on_date(CLASS_POSITIVE, 1, "2026-01-05")]
            + [sample_on_date(CLASS_NEGATIVE, index + 100, "2026-01-06") for index in range(4)]
            + [sample_on_date(CLASS_NEUTRAL, index + 200, "2026-01-06") for index in range(11)]
        )
        self.assertEqual(interleave_class_rows(rows, 11, "test"), [])

    def test_balanced_train_order_uses_class_labels_or_metadata(self):
        import random

        rows = (
            [sample_on_date(CLASS_POSITIVE, index, "2026-01-05") for index in range(2)]
            + [sample_on_date(CLASS_NEGATIVE, index + 100, "2026-01-05") for index in range(8)]
            + [sample_on_date(CLASS_NEUTRAL, index + 200, "2026-01-05") for index in range(22)]
        )
        rows[0]["class_label"] = rows[0]["metadata"].pop("label")
        order = threeclass_train._build_balanced_train_order(rows, 11, random.Random(11))
        labels = [threeclass_train._item_class_label(rows[index]) for index in order[:16]]
        self.assertEqual(labels, [CLASS_POSITIVE] + [CLASS_NEGATIVE] * 4 + [CLASS_NEUTRAL] * 11)
        self.assertEqual({rows[index]["metadata"]["anchor_date"] for index in order[:16]}, {"2026-01-05"})

    def test_balanced_train_order_rejects_cross_day_completion(self):
        import random

        rows = (
            [sample_on_date(CLASS_POSITIVE, 1, "2026-01-05")]
            + [sample_on_date(CLASS_NEGATIVE, index + 100, "2026-01-06") for index in range(4)]
            + [sample_on_date(CLASS_NEUTRAL, index + 200, "2026-01-06") for index in range(11)]
        )
        with self.assertRaisesRegex(RuntimeError, "strict same-date"):
            threeclass_train._build_balanced_train_order(rows, 11, random.Random(11))

    def test_training_defaults_use_requested_ratio_eval_size_and_learning_rate(self):
        args = threeclass_train.build_parser().parse_args([])
        self.assertEqual(args.data_dir, Path("blackbox_finetune_threeclass/data_xlong_p1_n4_u11"))
        self.assertEqual(args.eval_max_samples, 1500)
        self.assertEqual(args.learning_rate, 5e-6)

    def test_training_dataset_builder_defaults_to_all_rows_as_train(self):
        args = build_dataset_parser().parse_args([])
        self.assertEqual(args.output_split, "train")

    def test_validation_dataset_builder_writes_all_rows_as_test(self):
        with patch("blackbox_finetune_threeclass.build_validation_dataset.build_threeclass_dataset") as builder:
            build_validation_main(["--output-dir", "validation-dir"])
        self.assertEqual(builder.call_args.kwargs["output_split"], "test")

    def test_threeclass_train_accepts_checkpoint_eval_data_dir(self):
        args = threeclass_train.build_parser().parse_args(
            ["--checkpoint-eval-data-dir", "validation-dir"]
        )
        self.assertEqual(args.checkpoint_eval_data_dir, Path("validation-dir"))

    def test_threeclass_main_passes_recall60_required_training_args(self):
        with patch.object(threeclass_train, "prepare_rtx3060"), patch.object(
            threeclass_train.base_train,
            "train_recall60_lora",
        ) as train_lora:
            threeclass_train.main(
                [
                    "--data-dir",
                    "train-dir",
                    "--checkpoint-eval-data-dir",
                    "validation-dir",
                    "--output-dir",
                    "out-dir",
                    "--no-auto-resume",
                ]
            )
        kwargs = train_lora.call_args.kwargs
        self.assertEqual(kwargs["checkpoint_eval_data_dir"], Path("validation-dir"))
        self.assertIsNone(kwargs["initial_adapter_dir"])
        self.assertEqual(kwargs["evaluation_threshold_position"], 0.2)
        self.assertEqual(kwargs["positive_loss_weight"], 1.0)
        self.assertEqual(kwargs["negative_loss_weight"], 1.0)
        self.assertEqual(kwargs["high_score_positive_bonus"], 0.0)
        self.assertEqual(kwargs["high_score_positive_position"], 0.0)
        self.assertFalse(kwargs["fp_dynamic_penalty"])
        self.assertEqual(kwargs["fp_penalty_weight"], 0.0)

    def test_threeclass_training_summary_hides_binary_only_parameters(self):
        summary = threeclass_train._threeclass_training_run_summary(
            train_rows=10,
            valid_rows=2,
            checkpoint_eval_data_dir=Path("validation-dir"),
            total_updates=3,
            start_update=0,
            batch_size=1,
            gradient_accumulation_steps=32,
            train_seed=937498347,
            learning_rate=5e-6,
            weight_decay=0.05,
            max_grad_norm=1.0,
            lora_rank=8,
            lora_dropout=0.3,
            max_seq_length=3072,
            on_the_fly_tokenize=True,
            checkpoint_every=50,
            evaluation_threshold=0.48,
            evaluation_threshold_position=0.2,
            evaluation_precision_top_k=10,
            evaluation_precision_threshold=0.4,
            evaluation_max_samples=750,
        )
        self.assertIn("three-class LoRA train", summary)
        self.assertIn("high_score_positive_reward=", summary)
        self.assertNotIn("fp_penalty_weight", summary)
        self.assertNotIn("fp_threshold_ema_alpha", summary)

    def test_high_score_ema_can_be_disabled_by_argument(self):
        args = threeclass_train.build_parser().parse_args(["--no-high-score-ema"])
        self.assertFalse(args.high_score_ema)

    def test_three_class_answers_are_compact(self):
        self.assertEqual(label_answer(CLASS_POSITIVE), '{"c":"positive"}')
        self.assertEqual(label_answer(CLASS_NEGATIVE), '{"c":"negative"}')
        self.assertEqual(label_answer(CLASS_NEUTRAL), '{"c":"neutral"}')
        messages = compact_messages_from_sample(sample(CLASS_NEUTRAL, 1))
        self.assertEqual(messages[-1]["content"], '{"c":"neutral"}')

    def test_initial_binary_adapter_starts_at_update_zero(self):
        with TemporaryDirectory() as directory:
            adapter_dir = Path(directory) / "update-003200"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            initial_path = threeclass_train._validate_initial_binary_adapter(adapter_dir)
            original = threeclass_train.base_train._checkpoint_update
            self.assertEqual(
                threeclass_train._checkpoint_update_with_initial(adapter_dir, initial_path, original),
                0,
            )
            self.assertEqual(
                threeclass_train._checkpoint_update_with_initial(
                    Path("checkpoints/update-000200"),
                    initial_path,
                    original,
                ),
                200,
            )

    def test_initial_binary_adapter_requires_adapter_config(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                threeclass_train._validate_initial_binary_adapter(Path(directory))

    def test_initial_binary_adapter_argument_is_available(self):
        args = threeclass_train.build_parser().parse_args(
            ["--initial-binary-adapter-dir", "binary-adapter"]
        )
        self.assertEqual(args.initial_binary_adapter_dir, Path("binary-adapter"))

    def test_asymmetric_loss_defaults_are_configurable(self):
        args = threeclass_train.build_parser().parse_args([])
        self.assertEqual(args.positive_ce_weight, 2.0)
        self.assertEqual(args.negative_ce_weight, 1.0)
        self.assertEqual(args.neutral_ce_weight, 0.5)
        self.assertEqual(args.fp_loss_weight, 1.0)
        self.assertEqual(args.neutral_fp_loss_weight, 0.3)
        self.assertEqual(args.rank_loss_weight, 0.5)
        self.assertEqual(args.rank_margin, 0.2)
        self.assertEqual(args.neutral_rank_loss_weight, 0.2)
        self.assertEqual(args.neutral_rank_margin, 0.05)
        self.assertEqual(args.positive_high_score_loss_weight, 1.0)
        self.assertEqual(args.positive_high_score_margin, 0.0)
        self.assertTrue(args.high_score_ema)
        self.assertEqual(args.high_score_ema_alpha, 0.02)
        self.assertEqual(args.high_score_cutoff_position, 0.6)
        self.assertEqual(args.high_score_positive_bonus, 1.0)
        self.assertEqual(args.high_score_positive_bonus_scale, 0.05)
        self.assertEqual(args.high_score_positive_bonus_max_multiplier, 8.0)
        self.assertEqual(args.high_score_negative_penalty_weight, 1.0)
        self.assertEqual(args.high_score_neutral_penalty_weight, 0.5)
        self.assertTrue(args.fp_dynamic_penalty)

    def test_threeclass_extra_training_state_preserves_high_score_cutoff(self):
        threeclass_train._load_extra_training_state({"high_score_cutoff": -0.75})

        state = threeclass_train._extra_training_state()
        threeclass_train._load_extra_training_state({"high_score_cutoff": None})

        self.assertAlmostEqual(state["high_score_cutoff"], -0.75)
        self.assertIsNone(threeclass_train._extra_training_state()["high_score_cutoff"])

    def test_threeclass_tokenization_preserves_class_label(self):
        class FakeTokenizer:
            eos_token = "<eos>"
            eos_token_id = 0

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "prompt"

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1, 2] if text == "prompt" else [3, 0]}

        item = threeclass_train._tokenize_threeclass_row(
            FakeTokenizer(),
            sample(CLASS_NEGATIVE, 1),
            32,
        )
        self.assertEqual(item["class_label"], CLASS_NEGATIVE)

    def test_negative_auxiliary_losses_penalize_preferred_positive_answer(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this environment")

        threeclass_train._configure_asymmetric_loss(2.0, 1.0, 0.5, 1.0, 0.3, 0.5, 0.2, 0.2, 0.05, 1.0, 0.0, True, 0.02, 0.8, 1.0, 0.05, 8.0, 1.0, 0.5)
        low_fp, low_rank = threeclass_train._negative_auxiliary_losses(
            torch.tensor([0.0]),
            torch.tensor([2.0]),
        )
        high_fp, high_rank = threeclass_train._negative_auxiliary_losses(
            torch.tensor([2.0]),
            torch.tensor([0.0]),
        )
        self.assertLess(float(low_fp), float(high_fp))
        self.assertEqual(float(low_rank), 0.0)
        self.assertGreater(float(high_rank), 2.0)

    def test_negative_ranking_loss_uses_margin_plus_negative_minus_positive_nll(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this environment")

        threeclass_train._configure_asymmetric_loss(2.0, 1.0, 0.5, 1.0, 0.3, 0.5, 0.2, 0.2, 0.05, 1.0, 0.0, True, 0.02, 0.8, 1.0, 0.05, 8.0, 1.0, 0.5)
        _, rank = threeclass_train._negative_auxiliary_losses(
            torch.tensor([0.4]),
            torch.tensor([0.5]),
        )
        self.assertAlmostEqual(float(rank), 0.1, places=6)

    def test_neutral_false_positive_loss_penalizes_preferred_positive_answer(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this environment")

        low_fp = threeclass_train._neutral_false_positive_loss(
            torch.tensor([0.0]),
            torch.tensor([2.0]),
        )
        high_fp = threeclass_train._neutral_false_positive_loss(
            torch.tensor([2.0]),
            torch.tensor([0.0]),
        )
        self.assertLess(float(low_fp), float(high_fp))

    def test_neutral_ranking_loss_uses_neutral_margin(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this environment")

        threeclass_train._configure_asymmetric_loss(2.0, 1.0, 0.5, 1.0, 0.3, 0.5, 0.2, 0.2, 0.05, 1.0, 0.0, True, 0.02, 0.8, 1.0, 0.05, 8.0, 1.0, 0.5)
        low_rank = threeclass_train._neutral_ranking_loss(
            torch.tensor([0.4]),
            torch.tensor([0.6]),
        )
        high_rank = threeclass_train._neutral_ranking_loss(
            torch.tensor([0.4]),
            torch.tensor([0.42]),
        )
        self.assertEqual(float(low_rank), 0.0)
        self.assertAlmostEqual(float(high_rank), 0.03, places=6)

    def test_positive_high_score_reward_increases_with_score(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this environment")

        threeclass_train._configure_asymmetric_loss(
            2.0,
            1.0,
            0.5,
            1.0,
            0.3,
            0.5,
            0.2,
            0.2,
            0.05,
            1.0,
            0.0,
            True,
            0.02,
            0.8,
            2.0,
            0.05,
            8.0,
            1.0,
            0.5,
        )
        rewards = threeclass_train._positive_high_score_rewards(
            torch.tensor([0.50, 0.55, 0.70]),
            torch.tensor(0.50),
        )
        self.assertGreater(float(rewards[1]), float(rewards[0]))
        self.assertEqual(float(rewards[2]), 8.0)

    def test_probabilities_are_normalized_and_follow_loss(self):
        probabilities = probabilities_from_losses({CLASS_NEGATIVE: 2.0, CLASS_NEUTRAL: 1.0, CLASS_POSITIVE: 0.5})
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities[CLASS_POSITIVE], probabilities[CLASS_NEUTRAL])
        self.assertGreater(probabilities[CLASS_NEUTRAL], probabilities[CLASS_NEGATIVE])

    def test_selection_score_defaults_penalize_negative_probability(self):
        self.assertAlmostEqual(selection_score(0.5, 0.4, 0.1), 0.3)
        self.assertAlmostEqual(selection_score(0.5, 0.4, 0.1, 0.5, 1.0), 0.2)

    def test_next_eval_threshold_uses_top_twenty_percent_position(self):
        average, maximum, position, threshold = threeclass_train._next_selection_score_threshold(
            [0.1, 0.3, 0.5],
            current_threshold=0.0,
            top_ratio=0.2,
        )
        self.assertAlmostEqual(average, 0.3)
        self.assertAlmostEqual(maximum, 0.5)
        self.assertAlmostEqual(position, 0.8)
        self.assertAlmostEqual(threshold, 0.46)

    def test_prediction_weight_defaults_are_configurable(self):
        args = build_predict_parser().parse_args(["--date", "20260612"])
        self.assertEqual(args.negative_weight, 0.5)
        self.assertEqual(args.neutral_weight, 0.0)
        self.assertEqual(args.positive_threshold, 0.0)

    def test_prediction_ranking_uses_score_without_threshold_filter(self):
        ranked = rank_predictions(
            [
                {"SCode": "000001", "SelectionScore": -0.2, "PositiveProbability": 0.1},
                {"SCode": "000002", "SelectionScore": 0.3, "PositiveProbability": 0.4},
                {"SCode": "000003", "SelectionScore": -0.1, "PositiveProbability": 0.2},
            ],
            3,
        )
        self.assertEqual(ranked["SCode"].tolist(), ["000002", "000003", "000001"])

    def test_metrics_include_confusion_and_positive_precision(self):
        rows = [
            {
                "actual_label": CLASS_POSITIVE,
                "predicted_label": CLASS_POSITIVE,
                "positive_probability": 0.9,
                "selection_score": 0.3,
            },
            {
                "actual_label": CLASS_NEGATIVE,
                "predicted_label": CLASS_POSITIVE,
                "positive_probability": 0.8,
                "selection_score": 0.1,
            },
            {
                "actual_label": CLASS_NEUTRAL,
                "predicted_label": CLASS_NEUTRAL,
                "positive_probability": 0.1,
                "selection_score": 0.2,
            },
        ]
        summary = summarize_scored_rows(rows, (2,))
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["confusion_matrix"]["positive"]["positive"], 1)
        self.assertAlmostEqual(summary["positive_precision@2"], 0.5)

    def test_positive_precision_uses_selection_score_ranking(self):
        rows = [
            {
                "actual_label": CLASS_NEGATIVE,
                "predicted_label": CLASS_POSITIVE,
                "positive_probability": 0.95,
                "selection_score": 0.1,
            },
            {
                "actual_label": CLASS_POSITIVE,
                "predicted_label": CLASS_POSITIVE,
                "positive_probability": 0.70,
                "selection_score": 0.6,
            },
        ]
        summary = summarize_scored_rows(rows, (1,))
        self.assertEqual(summary["positive_precision@1"], 1.0)

    def test_checkpoint_top50_rows_keep_three_probabilities_and_actual_class(self):
        rows = [
            {
                "scode": "000001",
                "actual_class": "negative",
                "positive_probability": 0.8,
                "neutral_probability": 0.1,
                "negative_probability": 0.1,
            },
            {
                "scode": "000002",
                "actual_class": "positive",
                "positive_probability": 0.9,
                "neutral_probability": 0.05,
                "negative_probability": 0.05,
            },
        ]
        top_rows = positive_probability_top_rows(rows, 50)
        self.assertEqual(top_rows[0]["scode"], "000002")
        self.assertEqual(top_rows[0]["rank"], 1)
        self.assertEqual(top_rows[0]["actual_class"], "positive")
        self.assertIn("neutral_probability", top_rows[0])
        self.assertIn("negative_probability", top_rows[0])

    def test_selection_score_top_rows_use_selection_score_order(self):
        rows = [
            {"scode": "000001", "positive_probability": 0.9, "selection_score": 0.2},
            {"scode": "000002", "positive_probability": 0.7, "selection_score": 0.5},
        ]
        top_rows = selection_score_top_rows(rows, 50)
        self.assertEqual(top_rows[0]["scode"], "000002")
        self.assertEqual(top_rows[0]["rank"], 1)

    def test_checkpoint_evaluation_writes_positive_probability_top50(self):
        class FakeModel:
            training = True

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return messages[-1]["content"]

        rows = [sample(CLASS_POSITIVE, 1), sample(CLASS_NEGATIVE, 2)]
        predictions = [
            {
                "label_id": CLASS_POSITIVE,
                "label": "positive",
                "positive_probability": 0.8,
                "neutral_probability": 0.1,
                "negative_probability": 0.1,
            },
            {
                "label_id": CLASS_NEGATIVE,
                "label": "negative",
                "positive_probability": 0.2,
                "neutral_probability": 0.2,
                "negative_probability": 0.6,
            },
        ]
        with TemporaryDirectory() as directory, patch.object(
            threeclass_train,
            "score_prediction",
            side_effect=predictions,
        ):
            result = threeclass_train._evaluate_training_checkpoint(
                model=FakeModel(),
                tokenizer=FakeTokenizer(),
                rows=rows,
                output_dir=Path(directory),
                update=100,
                total_updates=1000,
                progress=0.1,
                trained_epochs=0.03,
                threshold=0.0,
                threshold_position=0.2,
                max_samples=0,
                max_seq_length=3072,
                precision_top_k=10,
                precision_threshold=0.4,
            )
            top = result["positive_probability_top50"]
            selection_top = result["selection_score_top50"]
            self.assertEqual(top[0]["actual_class"], "positive")
            self.assertEqual(top[0]["positive_probability"], 0.8)
            self.assertEqual(top[0]["neutral_probability"], 0.1)
            self.assertEqual(top[0]["negative_probability"], 0.1)
            self.assertAlmostEqual(top[0]["selection_score"], 0.75)
            self.assertEqual(selection_top[0]["actual_class"], "positive")
            self.assertAlmostEqual(selection_top[0]["selection_score"], 0.75)
            self.assertAlmostEqual(result["average_selection_score"], 0.325)
            self.assertAlmostEqual(result["max_selection_score"], 0.75)
            self.assertAlmostEqual(result["average_positive_probability"], 0.325)
            self.assertAlmostEqual(result["max_positive_probability"], 0.75)
            self.assertAlmostEqual(result["threshold_position"], 0.8)
            self.assertAlmostEqual(result["next_threshold"], 0.665)
            self.assertEqual(result["training_parameters"]["rank_loss_weight"], threeclass_train._RANK_LOSS_WEIGHT)
            self.assertEqual(result["training_parameters"]["neutral_rank_margin"], threeclass_train._NEUTRAL_RANK_MARGIN)
            self.assertEqual(result["evaluation_parameters"]["negative_weight"], 0.5)
            self.assertEqual(result["evaluation_parameters"]["eval_precision_top_k"], 10)
            self.assertTrue(list(Path(directory).glob("eval-update-000100-*.json")))


if __name__ == "__main__":
    unittest.main()
