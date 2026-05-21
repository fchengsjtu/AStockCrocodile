import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from blackbox_finetune_recall30 import build_dataset, build_validation_dataset, common, evaluate, gpu, predict_day, train


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

    def test_train_evaluate_predict_default_to_cuda_zero(self):
        self.assertEqual(train.build_parser().parse_args([]).cuda_device, "0")
        self.assertEqual(evaluate.build_parser().parse_args([]).cuda_device, "0")
        self.assertEqual(predict_day.build_parser().parse_args(["--date", "20260514"]).cuda_device, "0")

    def test_gpu_diagnose_parser_defaults_to_cuda_zero(self):
        args = gpu.build_parser().parse_args([])

        self.assertEqual(args.cuda_device, "0")
        self.assertFalse(args.allow_non_rtx3060)

    def test_train_no_4bit_kept_for_script_compatibility(self):
        args = train.build_parser().parse_args(["--no-4bit"])

        self.assertTrue(args.no_4bit)

    def test_train_stability_defaults(self):
        args = train.build_parser().parse_args([])

        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.max_grad_norm, 0.5)
        self.assertEqual(args.checkpoint_every, 1000)
        self.assertEqual(args.nonfinite_patience, 20)
        self.assertEqual(args.nonfinite_skip_limit, 100)
        self.assertEqual(args.nonfinite_backoff_every, 10)
        self.assertEqual(args.lr_backoff_factor, 0.5)
        self.assertEqual(args.min_learning_rate, 1e-6)
        self.assertEqual(args.oom_patience, 20)
        self.assertFalse(args.no_auto_resume)

    def test_format_duration_for_progress_log(self):
        self.assertEqual(train._format_duration(65), "01:05")
        self.assertEqual(train._format_duration(3661), "01:01:01")

    def test_latest_checkpoint_picks_highest_update(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ["update-000010", "update-000200", "bad-name"]:
                checkpoint = root / "checkpoints" / name
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")

            self.assertEqual(train._latest_checkpoint(root).name, "update-000200")
            self.assertEqual(train._checkpoint_update(root / "checkpoints" / "update-000200"), 200)


if __name__ == "__main__":
    unittest.main()
