import json
import unittest
from collections import Counter
from datetime import date

import llm_surge_pattern_miner


class LlmSurgePatternMinerTests(unittest.TestCase):
    def test_parse_llm_patterns_keeps_only_known_features(self):
        response = json.dumps(
            {
                "patterns": [
                    {"name": "ok", "features": ["D_CLOSE_GT_MA5", "W_MA5_GT_MA13", "D_RET_5_GE_5"]},
                    {"name": "joined", "features": ["D_CLOSE_GT_MA5 && W_MA5_GT_MA13 && D_RET_5_GE_5"]},
                    {"name": "text", "pattern": "D_CLOSE_GT_MA5 && W_MA5_GT_MA13 && D_RET_5_GE_5"},
                    {"name": "unknown", "features": ["D_CLOSE_GT_MA5", "NOT_A_FEATURE"]},
                    {"name": "too_big", "features": ["A", "B", "C"]},
                ]
            }
        )

        patterns = llm_surge_pattern_miner.parse_llm_patterns(
            response,
            {"D_CLOSE_GT_MA5", "W_MA5_GT_MA13", "D_RET_5_GE_5", "A", "B", "C"},
            min_pattern_size=3,
            max_pattern_size=8,
        )

        self.assertEqual(patterns, [("D_CLOSE_GT_MA5", "D_RET_5_GE_5", "W_MA5_GT_MA13"), ("A", "B", "C")])

    def test_fallback_patterns_from_counts_uses_high_support(self):
        patterns = llm_surge_pattern_miner.fallback_patterns_from_counts(
            Counter({("A", "B"): 5, ("C",): 3, ("A", "B", "C"): 10}),
            min_pattern_size=3,
            max_pattern_size=8,
            limit=2,
        )

        self.assertEqual(patterns, [("A", "B", "C")])

    def test_fallback_patterns_from_counts_builds_from_single_features(self):
        patterns = llm_surge_pattern_miner.fallback_patterns_from_counts(
            Counter({("A",): 5, ("B",): 4, ("C",): 3, ("D",): 2}),
            min_pattern_size=3,
            max_pattern_size=8,
            limit=2,
        )

        self.assertEqual(patterns, [("A", "B", "C"), ("A", "B", "D")])

    def test_build_llm_prompt_contains_feature_summary(self):
        config = llm_surge_pattern_miner.LlmPatternConfig(
            test_start_date="20260101",
            test_end_date="20260430",
            train_start_date="20100101",
            train_end_date="20251231",
            stat_type="short_term_surge_3d_20pct",
            success_rates=(0.25, 0.5),
            min_sample_count=20,
            min_positive_support=5,
            min_pattern_size=3,
            max_pattern_size=8,
            daily_window=56,
            weekly_window=56,
            batch_size=40,
            model="deepseek-r1-distill-qwen-14b",
            candidate_count=20,
            top_features=10,
            top_pairs=10,
            training_mode=llm_surge_pattern_miner.TRAINING_MODE_SUMMARY,
            raw_sample_size=30,
            api_base_url="http://127.0.0.1:1234/v1",
            api_key_env="LOCAL_LLM_API_KEY",
            llm_response_file=None,
            output=None,
            save_db=False,
        )

        prompt = llm_surge_pattern_miner.build_llm_prompt(
            Counter({("D_CLOSE_GT_MA5",): 9}),
            Counter({("D_CLOSE_GT_MA5", "W_MA5_GT_MA13"): 6}),
            10,
            config,
            date(2010, 1, 1),
            date(2025, 12, 31),
            date(2026, 1, 1),
            date(2026, 4, 30),
        )

        self.assertIn("D_CLOSE_GT_MA5", prompt)
        self.assertIn("2026-04-30", prompt)
        self.assertIn('"candidate_count": 20', prompt)

    def test_build_raw_kline_prompt_contains_bars_and_allowed_features(self):
        config = llm_surge_pattern_miner.LlmPatternConfig(
            test_start_date="20260101",
            test_end_date="20260430",
            train_start_date="20100101",
            train_end_date="20251231",
            stat_type="short_term_surge_3d_20pct",
            success_rates=(0.25, 0.5),
            min_sample_count=20,
            min_positive_support=5,
            min_pattern_size=3,
            max_pattern_size=8,
            daily_window=55,
            weekly_window=55,
            batch_size=40,
            model="deepseek-r1-distill-qwen-14b",
            candidate_count=20,
            top_features=10,
            top_pairs=10,
            training_mode=llm_surge_pattern_miner.TRAINING_MODE_RAW_KLINE,
            raw_sample_size=3,
            api_base_url="http://127.0.0.1:1234/v1",
            api_key_env="LOCAL_LLM_API_KEY",
            llm_response_file=None,
            output=None,
            save_db=False,
        )
        samples = [
            {
                "scode": "000001",
                "selection_date": "2025-12-31",
                "daily_bars": [{"date": "2025-12-31", "close": 10.0}],
                "weekly_bars": [{"date": "2025-12-26", "close": 9.5}],
            }
        ]

        prompt = llm_surge_pattern_miner.build_raw_kline_llm_prompt(
            samples,
            Counter({("D_CLOSE_GT_MA5",): 3, ("D_CLOSE_GT_MA5", "W_MA5_GT_MA13"): 2}),
            config,
            date(2010, 1, 1),
            date(2025, 12, 31),
            date(2026, 1, 1),
            date(2026, 4, 30),
        )

        self.assertIn('"input_mode": "raw-kline"', prompt)
        self.assertIn('"daily_bars"', prompt)
        self.assertIn("D_CLOSE_GT_MA5", prompt)

    def test_estimate_positive_support_uses_min_single_support_for_large_patterns(self):
        support = llm_surge_pattern_miner.estimate_positive_support(
            ("A", "B", "C"),
            Counter({("A",): 9, ("B",): 7, ("C",): 5}),
            Counter(),
        )

        self.assertEqual(support, 5)

    def test_parser_defaults_to_local_llm_endpoint(self):
        parser = llm_surge_pattern_miner.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.model, "deepseek-r1-distill-qwen-14b")
        self.assertEqual(args.api_base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(args.api_key_env, "LOCAL_LLM_API_KEY")


if __name__ == "__main__":
    unittest.main()
