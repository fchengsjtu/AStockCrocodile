import unittest

import pandas as pd

import llm_pattern_selector


class LlmPatternSelectorTests(unittest.TestCase):
    def test_parse_pattern_text_splits_conjunction(self):
        pattern = llm_pattern_selector.parse_pattern_text("D_CLOSE_GT_MA5 && W_MA5_GT_MA13")

        self.assertEqual(pattern, ("D_CLOSE_GT_MA5", "W_MA5_GT_MA13"))

    def test_match_patterns_uses_feature_superset(self):
        patterns = pd.DataFrame(
            [
                {"PatternText": "A && B", "Pattern": ("A", "B"), "SuccessRate": 0.4},
                {"PatternText": "A && C", "Pattern": ("A", "C"), "SuccessRate": 0.5},
            ]
        )

        matched = llm_pattern_selector.match_patterns({"A", "B", "D"}, patterns)

        self.assertEqual(matched["PatternText"].tolist(), ["A && B"])

    def test_normalize_rate_accepts_percent_value(self):
        self.assertEqual(llm_pattern_selector.normalize_rate(35), 0.35)
        self.assertEqual(llm_pattern_selector.normalize_rate(0.35), 0.35)


if __name__ == "__main__":
    unittest.main()
