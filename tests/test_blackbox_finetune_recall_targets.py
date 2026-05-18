import importlib
import unittest


TARGETS = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]


class BlackboxFinetuneRecallTargetsTests(unittest.TestCase):
    def test_target_packages_have_expected_defaults(self):
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
                self.assertEqual(common.DEFAULT_TRAIN_START_DATE, "20110101")
                self.assertEqual(common.DEFAULT_TRAIN_END_DATE, "20251231")
                self.assertEqual(common.DEFAULT_VALIDATION_START_DATE, "20260101")
                self.assertEqual(common.DEFAULT_VALIDATION_END_DATE, "20260430")
                self.assertIn(package, str(common.DEFAULT_DATA_DIR))
                self.assertIn(package, str(common.DEFAULT_VALIDATION_DIR))
                self.assertIn(package, str(common.DEFAULT_OUTPUT_DIR))

                build_args = build_dataset.build_parser().parse_args([])
                validation_args = build_validation.build_parser().parse_args([])
                evaluate_args = evaluate.build_parser().parse_args([])
                train_args = train.build_parser().parse_args([])
                predict_args = predict_day.build_parser().parse_args(["--date", "20260514"])

                self.assertEqual(build_args.start_date, "20110101")
                self.assertEqual(build_args.end_date, "20251231")
                self.assertEqual(validation_args.start_date, "20260101")
                self.assertEqual(validation_args.end_date, "20260430")
                self.assertEqual(evaluate_args.min_positive_recall, target / 100)
                self.assertEqual(train_args.cuda_device, "0")
                self.assertEqual(predict_args.cuda_device, "0")


if __name__ == "__main__":
    unittest.main()
