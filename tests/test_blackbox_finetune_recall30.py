import unittest

from blackbox_finetune_recall30 import build_dataset, build_validation_dataset, common, evaluate


class BlackboxFinetuneRecall30Tests(unittest.TestCase):
    def test_default_dates_match_recall30_task(self):
        parser = build_dataset.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.start_date, "20110101")
        self.assertEqual(args.end_date, "20251231")
        self.assertEqual(args.output_dir, common.DEFAULT_DATA_DIR)

    def test_validation_defaults_match_holdout_period(self):
        parser = build_validation_dataset.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.start_date, "20260101")
        self.assertEqual(args.end_date, "20260430")
        self.assertEqual(args.output_dir, common.DEFAULT_VALIDATION_DIR)

    def test_evaluate_default_positive_recall_target_is_30_percent(self):
        parser = evaluate.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.min_positive_recall, 0.30)
        self.assertEqual(args.data_dir, common.DEFAULT_VALIDATION_DIR)


if __name__ == "__main__":
    unittest.main()
