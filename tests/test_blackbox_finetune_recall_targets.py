import importlib
from pathlib import Path
import unittest


TARGETS = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]


class BlackboxFinetuneRecallTargetsTests(unittest.TestCase):
    def test_target_packages_have_expected_defaults(self):
        output_dirs = set()
        train_seeds = set()
        for target in TARGETS:
            with self.subTest(target=target):
                package = f"blackbox_finetune_recall{target}"
                common = importlib.import_module(f"{package}.common")
                build_dataset = importlib.import_module(f"{package}.build_dataset")
                build_validation = importlib.import_module(f"{package}.build_validation_dataset")
                evaluate = importlib.import_module(f"{package}.evaluate")
                train = importlib.import_module(f"{package}.train")
                predict_day = importlib.import_module(f"{package}.predict_day")

                self.assertEqual(common.DEFAULT_MIN_POSITIVE_RECALL, target / 100)
                self.assertEqual(common.DEFAULT_TRAIN_SEED, 20260500 + target)
                expected_train_start = "20110101" if target == 80 else "20200101"
                self.assertEqual(common.DEFAULT_TRAIN_START_DATE, expected_train_start)
                expected_train_end = "20241231" if target == 80 else "20251231"
                self.assertEqual(common.DEFAULT_TRAIN_END_DATE, expected_train_end)
                self.assertEqual(common.DEFAULT_VALIDATION_START_DATE, "20260101")
                self.assertEqual(common.DEFAULT_VALIDATION_END_DATE, "20260430")
                self.assertEqual(common.DEFAULT_DATA_DIR, common.default_data_dir("long"))
                self.assertEqual(common.DEFAULT_VALIDATION_DIR, common.default_validation_dir("long"))
                self.assertIn(f"{package}\\data_no_partial_week_long", str(common.default_data_dir("long")))
                self.assertIn(f"{package}\\data_no_partial_week_short", str(common.default_data_dir("short")))
                self.assertIn(f"{package}\\data_no_partial_week_xlong", str(common.default_data_dir("xlong")))
                self.assertIn(f"{package}\\data_no_partial_week_xxlong", str(common.default_data_dir("xxlong")))
                self.assertIn(f"{package}\\data_evaluation_no_partial_week_long", str(common.default_validation_dir("long")))
                self.assertIn(package, str(common.DEFAULT_OUTPUT_DIR))
                self.assertIn(f"recall{target}-long-lora", str(common.default_output_dir("long")))
                self.assertIn(f"recall{target}-short-lora", str(common.default_output_dir("short")))
                self.assertIn(f"recall{target}-xlong-lora", str(common.default_output_dir("xlong")))
                self.assertIn(f"recall{target}-xxlong-lora", str(common.default_output_dir("xxlong")))
                output_dirs.add(str(common.DEFAULT_OUTPUT_DIR))
                train_seeds.add(common.DEFAULT_TRAIN_SEED)

                build_args = build_dataset.build_parser().parse_args([])
                validation_args = build_validation.build_parser().parse_args([])
                evaluate_args = evaluate.build_parser().parse_args([])
                train_args = train.build_parser().parse_args([])
                predict_args = predict_day.build_parser().parse_args(["--date", "20260514"])

                self.assertEqual(build_args.start_date, expected_train_start)
                self.assertEqual(build_args.end_date, expected_train_end)
                self.assertEqual(build_args.negative_ratio, 3.0)
                self.assertEqual(validation_args.start_date, "20260101")
                self.assertEqual(validation_args.end_date, "20260430")
                self.assertEqual(validation_args.negative_ratio, 3.0)
                self.assertEqual(evaluate_args.min_positive_recall, target / 100)
                self.assertEqual(train_args.train_seed, 20260500 + target)
                self.assertEqual(train_args.cuda_device, "0")
                expected_learning_rate = 2e-5 if target == 80 else 5e-6
                self.assertEqual(train_args.learning_rate, expected_learning_rate)
                self.assertEqual(train_args.weight_decay, 0.0)
                self.assertEqual(build_args.sample_mode, "long")
                self.assertEqual(validation_args.sample_mode, "long")
                self.assertEqual(predict_args.sample_mode, "long")
                self.assertEqual(common.sample_mode_config("xlong"), {"daily": 21, "weekly": 13, "monthly": 8, "max_seq_length": 3072})
                self.assertEqual(common.default_max_seq_length("xlong"), 3072)
                self.assertEqual(common.sample_mode_config("xxlong"), {"daily": 34, "weekly": 21, "monthly": 13, "max_seq_length": 4096})
                self.assertEqual(common.default_max_seq_length("xxlong"), 4096)
                self.assertEqual(build_dataset.build_parser().parse_args(["--sample-mode", "xlong"]).sample_mode, "xlong")
                self.assertEqual(build_validation.build_parser().parse_args(["--sample-mode", "xlong"]).sample_mode, "xlong")
                self.assertEqual(predict_day.build_parser().parse_args(["--date", "20260514", "--sample-mode", "xlong"]).sample_mode, "xlong")
                self.assertEqual(build_dataset.build_parser().parse_args(["--sample-mode", "xxlong"]).sample_mode, "xxlong")
                self.assertEqual(build_validation.build_parser().parse_args(["--sample-mode", "xxlong"]).sample_mode, "xxlong")
                self.assertEqual(predict_day.build_parser().parse_args(["--date", "20260514", "--sample-mode", "xxlong"]).sample_mode, "xxlong")
                self.assertEqual(train_args.max_seq_length, 2048)
                self.assertEqual(evaluate_args.max_seq_length, 2048)
                self.assertEqual(predict_args.max_seq_length, 2048)
                self.assertEqual(train_args.max_grad_norm, 0.5)
                self.assertEqual(train_args.lora_rank, 16)
                self.assertEqual(train_args.lora_dropout, 0.05)
                self.assertEqual(train_args.checkpoint_every, 500)
                self.assertEqual(train_args.nonfinite_patience, 20)
                if target == 80:
                    self.assertEqual(train_args.min_seq_length_on_oom, 512)
                    self.assertEqual(train_args.oom_shrink_factor, 0.5)
                else:
                    self.assertFalse(hasattr(train_args, "min_seq_length_on_oom"))
                    self.assertFalse(hasattr(train_args, "oom_shrink_factor"))
                self.assertEqual(train_args.nonfinite_skip_limit, 100)
                self.assertEqual(train_args.nonfinite_backoff_every, 10)
                self.assertEqual(train_args.lr_backoff_factor, 0.5)
                self.assertEqual(train_args.min_learning_rate, 1e-6)
                self.assertEqual(predict_args.cuda_device, "0")

                predict_source = Path(package, "predict_day.py").read_text(encoding="utf-8")
                self.assertIn("_sample_windows_are_valid", predict_source)
                self.assertIn("load_abnormal_symbols", predict_source)
                self.assertIn("skipped_by_sample_rule", predict_source)

        self.assertEqual(len(output_dirs), len(TARGETS))
        self.assertEqual(len(train_seeds), len(TARGETS))


if __name__ == "__main__":
    unittest.main()
