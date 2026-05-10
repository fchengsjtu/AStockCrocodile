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
                    {"name": "ok", "features": ["D_CLOSE_GT_MA5", "W_MA5_GT_MA13"]},
                    {"name": "unknown", "features": ["D_CLOSE_GT_MA5", "NOT_A_FEATURE"]},
                    {"name": "too_big", "features": ["A", "B", "C"]},
                ]
            }
        )

        patterns = llm_surge_pattern_miner.parse_llm_patterns(
            response,
            {"D_CLOSE_GT_MA5", "W_MA5_GT_MA13", "A", "B", "C"},
            max_pattern_size=2,
        )

        self.assertEqual(patterns, [("D_CLOSE_GT_MA5", "W_MA5_GT_MA13")])

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
            max_pattern_size=2,
            daily_window=56,
            weekly_window=56,
            batch_size=40,
            model="deepseek-chat",
            candidate_count=20,
            top_features=10,
            top_pairs=10,
            api_base_url="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_API_KEY",
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


if __name__ == "__main__":
    unittest.main()
