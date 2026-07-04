from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from blackbox_negative_reshuffle.core import (
    NEGATIVE_KIND_DROP6,
    NEGATIVE_KIND_NEUTRAL,
    load_source_metadata_from_paths,
    load_source_metadata,
    plan_negative_kind_counts,
    reshuffle_split,
    reshuffle_split_by_negative_kind,
    row_key,
    write_jsonl,
)
from blackbox_negative_reshuffle.run import (
    build_parser,
    infer_dataset_settings,
    load_cached_negative_scores,
)


def sample(code: str, label: int) -> dict:
    return {
        "metadata": {
            "scode": code,
            "anchor_date": "2026-01-01",
            "label": label,
        }
    }


class BlackboxNegativeReshuffleTests(unittest.TestCase):
    def test_duplicate_train_path_keys_are_read_as_train_and_eval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "model"
            train_dir = root / "train"
            eval_dir = root / "eval"
            model_dir.mkdir()
            for name in ("train.jsonl", "test.jsonl", "all.jsonl"):
                write_jsonl(train_dir / name, [sample("000001", 1)])
            write_jsonl(eval_dir / "test.jsonl", [sample("000002", 0)])
            eval_json = model_dir / "eval-1.json"
            eval_json.write_text(
                "{"
                f"\"original_train_dataset_path\":{json.dumps(str(train_dir))},"
                f"\"original_train_dataset_path\":{json.dumps(str(eval_dir))}"
                "}",
                encoding="utf-8",
            )

            metadata = load_source_metadata(model_dir)

            self.assertEqual(metadata.training_dataset_dir, train_dir.resolve())
            self.assertEqual(metadata.evaluation_dataset_dir, eval_dir.resolve())

    def test_explicit_dataset_dirs_override_eval_json_dataset_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "model"
            old_train_dir = root / "old_train"
            old_eval_dir = root / "old_eval"
            train_dir = root / "train"
            eval_dir = root / "eval"
            model_dir.mkdir()
            for directory in (old_train_dir, train_dir):
                for name in ("train.jsonl", "test.jsonl", "all.jsonl"):
                    write_jsonl(directory / name, [sample("000001", 1)])
            for directory in (old_eval_dir, eval_dir):
                write_jsonl(directory / "test.jsonl", [sample("000002", 0)])
            eval_json = model_dir / "eval-1.json"
            eval_json.write_text(
                json.dumps(
                    {
                        "original_train_dataset_path": str(old_train_dir),
                        "original_eval_dataset_path": str(old_eval_dir),
                    }
                ),
                encoding="utf-8",
            )

            metadata = load_source_metadata_from_paths(model_dir, train_dir, eval_dir, eval_json)

            self.assertEqual(metadata.evaluation_json, eval_json.resolve())
            self.assertEqual(metadata.training_dataset_dir, train_dir.resolve())
            self.assertEqual(metadata.evaluation_dataset_dir, eval_dir.resolve())

    def test_reshuffle_keeps_high_scores_and_avoids_excluded_rows(self):
        positives = [sample("P1", 1)]
        negatives = [sample(f"N{index}", 0) for index in range(6)]
        source = positives + negatives[:4]
        scored = [(1.0 - index / 10.0, row) for index, row in enumerate(negatives)]

        output, selected_keys, stats = reshuffle_split(
            source,
            scored,
            negatives[2:],
            keep_count=2,
            rng=__import__("random").Random(7),
            excluded_keys={row_key(negatives[5])},
        )

        output_negative_keys = {row_key(row) for row in output if row["metadata"]["label"] == 0}
        self.assertIn(row_key(negatives[0]), output_negative_keys)
        self.assertIn(row_key(negatives[1]), output_negative_keys)
        self.assertNotIn(row_key(negatives[5]), output_negative_keys)
        self.assertEqual(selected_keys, output_negative_keys)
        self.assertEqual(stats["retained_hard_negatives"], 2)
        self.assertEqual(stats["random_replacements"], 2)

    def test_reshuffle_can_reduce_to_target_negative_count(self):
        positives = [sample("P1", 1), sample("P2", 1)]
        negatives = [sample(f"N{index}", 0) for index in range(10)]
        source = positives + negatives[:8]
        scored = [(1.0 - index / 10.0, row) for index, row in enumerate(negatives)]

        output, _, stats = reshuffle_split(
            source,
            scored,
            negatives[3:],
            keep_count=3,
            rng=__import__("random").Random(7),
            target_negative_count=5,
        )

        self.assertEqual(sum(row["metadata"]["label"] == 1 for row in output), 2)
        self.assertEqual(sum(row["metadata"]["label"] == 0 for row in output), 5)
        self.assertEqual(stats["source_negative_count"], 8)
        self.assertEqual(stats["negative_count"], 5)
        self.assertEqual(stats["retained_hard_negatives"], 3)
        self.assertEqual(stats["random_replacements"], 2)

    def test_negative_kind_plan_targets_drop6_three_parts_and_keeps_one_part(self):
        plan = plan_negative_kind_counts(
            positive_count=10,
            target_negative_count=90,
            drop6_target_ratio=3.0,
            drop6_keep_ratio=1.0,
            neutral_keep_ratio=1.0,
        )

        self.assertEqual(plan.target_drop6_count, 30)
        self.assertEqual(plan.target_neutral_count, 60)
        self.assertEqual(plan.keep_drop6_count, 10)
        self.assertEqual(plan.keep_neutral_count, 10)

    def test_reshuffle_by_negative_kind_keeps_top_drop6_and_neutral_then_refills(self):
        positives = [sample("P1", 1), sample("P2", 1)]
        drop6_negatives = [sample(f"D{index}", 0) for index in range(4)]
        neutral_negatives = [sample(f"U{index}", 0) for index in range(8)]
        drop6_pool = [sample(f"RD{index}", 0) for index in range(10)]
        neutral_pool = [sample(f"RU{index}", 0) for index in range(20)]
        for row in drop6_pool:
            row["metadata"]["negative_kind"] = NEGATIVE_KIND_DROP6
        for row in neutral_pool:
            row["metadata"]["negative_kind"] = NEGATIVE_KIND_NEUTRAL
        source = positives + drop6_negatives + neutral_negatives
        scored_drop6 = [(1.0 - index / 10.0, row) for index, row in enumerate(drop6_negatives)]
        scored_neutral = [(1.0 - index / 10.0, row) for index, row in enumerate(neutral_negatives)]
        plan = plan_negative_kind_counts(2, 18)

        output, selected_keys, stats = reshuffle_split_by_negative_kind(
            source,
            scored_drop6,
            scored_neutral,
            drop6_pool,
            neutral_pool,
            plan,
            rng=__import__("random").Random(7),
        )

        output_negative_keys = {row_key(row) for row in output if row["metadata"]["label"] == 0}
        self.assertEqual(len(output_negative_keys), 18)
        self.assertEqual(selected_keys, output_negative_keys)
        self.assertIn(row_key(drop6_negatives[0]), output_negative_keys)
        self.assertIn(row_key(drop6_negatives[1]), output_negative_keys)
        self.assertIn(row_key(neutral_negatives[0]), output_negative_keys)
        self.assertIn(row_key(neutral_negatives[1]), output_negative_keys)
        self.assertEqual(stats["drop6"]["target_count"], 6)
        self.assertEqual(stats["drop6"]["retained_hard_negatives"], 2)
        self.assertEqual(stats["drop6"]["random_replacements"], 4)
        self.assertEqual(stats["neutral"]["target_count"], 12)
        self.assertEqual(stats["neutral"]["retained_hard_negatives"], 2)
        self.assertEqual(stats["neutral"]["random_replacements"], 10)

    def test_dataset_settings_are_inferred_from_original_rows(self):
        rows = [
            {
                "metadata": {
                    "scode": "000001",
                    "anchor_date": "2023-01-03",
                    "label": 0,
                    "sample_mode": "xlong",
                }
            },
            {
                "metadata": {
                    "scode": "000002",
                    "anchor_date": "2024-12-30",
                    "label": 1,
                    "sample_mode": "xlong",
                }
            },
        ]

        mode, start_date, end_date = infer_dataset_settings(rows, None)

        self.assertEqual(mode, "xlong")
        self.assertEqual(start_date.isoformat(), "2023-01-03")
        self.assertEqual(end_date.isoformat(), "2024-12-30")

    def test_database_refill_defaults_to_twenty_attempts(self):
        args = build_parser().parse_args(["--model-dir", "model", "--target-negative-ratio", "5"])

        self.assertEqual(args.database_max_attempts, 20)
        self.assertEqual(args.keep_ratio, 0.30)
        self.assertEqual(args.target_negative_ratio, 5.0)

    def test_parser_accepts_explicit_dataset_dirs(self):
        args = build_parser().parse_args(
            [
                "--model-dir",
                "model",
                "--train-dataset-dir",
                "training",
                "--eval-dataset-dir",
                "evaluation",
            ]
        )

        self.assertEqual(str(args.train_dataset_dir), "training")
        self.assertEqual(str(args.eval_dataset_dir), "evaluation")

    def test_cached_negative_scores_are_reused_in_rank_order(self):
        with tempfile.TemporaryDirectory() as temp:
            score_path = Path(temp) / "negative_scores.jsonl"
            negatives = [sample("000001", 0), sample("000002", 0)]
            write_jsonl(
                score_path,
                [
                    {"scode": "000002", "anchor_date": "2026-01-01", "score": 0.9},
                    {"scode": "000001", "anchor_date": "2026-01-01", "score": 0.2},
                ],
            )

            scored = load_cached_negative_scores(score_path, negatives)

        self.assertEqual([row["metadata"]["scode"] for _, row in scored], ["000002", "000001"])
        self.assertEqual([score for score, _ in scored], [0.9, 0.2])

    def test_cached_negative_scores_rank_missing_rows_last(self):
        with tempfile.TemporaryDirectory() as temp:
            score_path = Path(temp) / "negative_scores.jsonl"
            negatives = [sample("000001", 0), sample("000002", 0)]
            write_jsonl(
                score_path,
                [
                    {"scode": "000001", "anchor_date": "2026-01-01", "score": 0.2},
                ],
            )

            scored = load_cached_negative_scores(score_path, negatives)

        self.assertEqual([row["metadata"]["scode"] for _, row in scored], ["000001", "000002"])
        self.assertEqual(scored[-1][0], float("-inf"))


if __name__ == "__main__":
    unittest.main()
