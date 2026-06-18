from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from blackbox_negative_reshuffle.core import (
    load_source_metadata,
    reshuffle_split,
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
        args = build_parser().parse_args(["--model-dir", "model"])

        self.assertEqual(args.database_max_attempts, 20)
        self.assertEqual(args.keep_ratio, 0.30)

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
