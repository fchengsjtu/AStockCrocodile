import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import date
from unittest.mock import patch

from blackbox_finetune import build_dataset as base_build_dataset
from blackbox_finetune.common import SampleEvent
from blackbox_finetune_recall60 import build_dataset, build_validation_dataset, common, evaluate, predict_day, train

PROJECT_ENV_KEYS = [
    "SAMPLE_MODE",
    "SAMPLE_BOTTOM_BAND_RATIO",
    "TRAIN_START_DATE",
    "TRAIN_END_DATE",
    "VALIDATION_START_DATE",
    "VALIDATION_END_DATE",
    "TEST_START_DATE",
    "TEST_END_DATE",
    "NEGATIVE_RATIO",
    "RECALL_TARGET",
    "MIN_POSITIVE_RECALL",
    "PRECISION_TOP_K",
    "PRECISION_THRESHOLD",
    "MIN_PRECISION_AT_20",
    "PRECISION_AT_20_TARGET",
    "EVAL_SAMPLE_METHOD",
    "EVAL_THRESHOLD_POSITION",
    "POSITIVE_LOSS_WEIGHT",
    "NEGATIVE_LOSS_WEIGHT",
    "HIGH_SCORE_POSITIVE_BONUS",
    "HIGH_SCORE_POSITIVE_POSITION",
    "FP_DYNAMIC_PENALTY",
    "FP_PENALTY_WEIGHT",
    "FP_THRESHOLD_EMA_ALPHA",
    "FP_THRESHOLD_MIN",
    "FP_THRESHOLD_MAX",
]


@contextmanager
def without_project_env():
    saved = {key: os.environ.pop(key, None) for key in PROJECT_ENV_KEYS}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def daily(day, open_, high, low, close, volume=100.0, amount=1000.0):
    return {
        "date": day,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
        "ma5": 0.0,
        "ma13": 0.0,
        "ma34": 0.0,
        "ma55": 0.0,
    }


class BlackboxFinetuneRecall60Tests(unittest.TestCase):
    def test_next_evaluation_threshold_uses_twenty_percent_position(self):
        average_probability, max_probability, next_threshold = train._next_evaluation_threshold(
            [0.2, 0.4, 0.6],
            current_threshold=0.48,
        )

        self.assertAlmostEqual(average_probability, 0.4)
        self.assertAlmostEqual(max_probability, 0.6)
        self.assertAlmostEqual(next_threshold, 0.44)

    def test_next_evaluation_threshold_position_is_configurable(self):
        _, _, next_threshold = train._next_evaluation_threshold(
            [0.2, 0.4, 0.6],
            current_threshold=0.48,
            threshold_position=0.8,
        )

        self.assertAlmostEqual(next_threshold, 0.56)

    def test_dynamic_fp_penalty_cutoff_uses_ema_and_bounds(self):
        updated = train._update_fp_penalty_cutoff(
            current_cutoff=0.50,
            next_threshold=0.60,
            max_probability=0.80,
            ema_alpha=0.2,
            minimum=0.45,
            maximum=0.65,
        )

        self.assertAlmostEqual(updated, 0.54)
        self.assertEqual(train._clamp_fp_penalty_cutoff(0.2, 0.45, 0.65), 0.45)
        self.assertEqual(train._clamp_fp_penalty_cutoff(0.9, 0.45, 0.65), 0.65)

    def test_high_scoring_negative_penalty_is_linear(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        penalty = train._high_scoring_negative_penalty(torch.tensor(0.45), 0.40)

        self.assertAlmostEqual(float(penalty), 0.05, places=6)

    def test_dynamic_fp_penalty_defaults_are_stronger(self):
        with without_project_env():
            args = train.build_parser().parse_args([])

        self.assertEqual(args.fp_penalty_weight, 1.0)
        self.assertEqual(args.fp_threshold_min, 0.40)

    def test_training_defaults_use_2020_to_2025(self):
        with without_project_env():
            args = build_dataset.build_parser().parse_args([])

        self.assertEqual(args.start_date, "20200101")
        self.assertEqual(args.end_date, "20251231")
        self.assertEqual(args.output_dir, common.default_data_dir("long"))
        self.assertIn("recall60_long", str(args.output_dir))
        self.assertEqual(args.negative_ratio, 5.0)

    def test_negative_ratio_default_can_come_from_environment(self):
        with patch.dict("os.environ", {"NEGATIVE_RATIO": "2.5"}):
            build_args = build_dataset.build_parser().parse_args([])
            validation_args = build_validation_dataset.build_parser().parse_args([])

        self.assertEqual(build_args.negative_ratio, 2.5)
        self.assertEqual(validation_args.negative_ratio, 2.5)

    def test_regularization_and_training_eval_args_can_come_from_environment(self):
        with patch.dict(
            "os.environ",
            {
                "LORA_RANK": "8",
                "LORA_DROPOUT": "0.10",
                "WEIGHT_DECAY": "0.01",
                "EVAL_THRESHOLD": "0.6",
                "EVAL_THRESHOLD_POSITION": "0.7",
                "EVAL_PRECISION_TOP_K": "12",
                "EVAL_PRECISION_THRESHOLD": "0.35",
                "EVAL_MAX_SAMPLES": "17",
                "POSITIVE_LOSS_WEIGHT": "2.5",
                "NEGATIVE_LOSS_WEIGHT": "0.8",
                "HIGH_SCORE_POSITIVE_BONUS": "1.5",
                "HIGH_SCORE_POSITIVE_POSITION": "0.75",
                "FP_DYNAMIC_PENALTY": "1",
                "FP_PENALTY_WEIGHT": "0.2",
                "FP_THRESHOLD_EMA_ALPHA": "0.3",
                "FP_THRESHOLD_MIN": "0.4",
                "FP_THRESHOLD_MAX": "0.9",
                "RECALL_TARGET": "80",
                "PRECISION_TOP_K": "20",
                "PRECISION_THRESHOLD": "0.30",
                "ON_THE_FLY_TOKENIZE": "1",
            },
        ):
            args = train.build_parser().parse_args([])
            eval_args = evaluate.build_parser().parse_args([])

        self.assertEqual(args.lora_rank, 8)
        self.assertEqual(args.lora_dropout, 0.10)
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.eval_every_epoch_fraction, 0.0)
        self.assertEqual(args.eval_threshold, 0.6)
        self.assertEqual(args.eval_threshold_position, 0.7)
        self.assertEqual(args.eval_precision_top_k, 12)
        self.assertEqual(args.eval_precision_threshold, 0.35)
        self.assertEqual(args.eval_max_samples, 17)
        self.assertEqual(args.positive_loss_weight, 2.5)
        self.assertEqual(args.negative_loss_weight, 0.8)
        self.assertEqual(args.high_score_positive_bonus, 1.5)
        self.assertEqual(args.high_score_positive_position, 0.75)
        self.assertTrue(args.fp_dynamic_penalty)
        self.assertEqual(args.fp_penalty_weight, 0.2)
        self.assertEqual(args.fp_threshold_ema_alpha, 0.3)
        self.assertEqual(args.fp_threshold_min, 0.4)
        self.assertEqual(args.fp_threshold_max, 0.9)
        self.assertTrue(args.on_the_fly_tokenize)
        self.assertIsNone(eval_args.min_positive_recall)
        self.assertEqual(eval_args.precision_top_k, 20)
        self.assertEqual(eval_args.precision_threshold, 0.30)
        self.assertIn("recall80", str(args.output_dir))

    def test_initial_adapter_dir_is_supported(self):
        args = train.build_parser().parse_args(["--initial-adapter-dir", "seed-adapter"])

        self.assertEqual(args.initial_adapter_dir, Path("seed-adapter"))

    def test_resolve_pretrained_source_validates_local_model_directory(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "Qwen2.5-0.5B-Instruct"
            model_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                common.resolve_pretrained_source(str(model_dir))
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(common.resolve_pretrained_source(str(model_dir)), str(model_dir))

    def test_precision_at_k_orders_by_positive_probability(self):
        scored = [
            {"actual_label": 0, "positive_probability": 0.95},
            {"actual_label": 1, "positive_probability": 0.90},
            {"actual_label": 1, "positive_probability": 0.80},
            {"actual_label": 0, "positive_probability": 0.70},
            {"actual_label": 1, "positive_probability": 0.60},
            {"actual_label": 0, "positive_probability": 0.50},
        ]

        result = common.precision_at_k(scored, (5, 10, 20, 100))

        self.assertEqual(result["precision@5"], 0.6)
        self.assertEqual(result["precision@10"], 0.5)
        self.assertEqual(result["precision@20"], 0.5)
        self.assertEqual(result["precision@100"], 0.5)

    def test_evaluation_summary_counts_predicted_labels(self):
        scored = [
            {"actual_label": 1, "predicted_label": 1, "positive_probability": 0.90},
            {"actual_label": 0, "predicted_label": 1, "positive_probability": 0.80},
            {"actual_label": 0, "predicted_label": 0, "positive_probability": 0.20},
            {"actual_label": 1, "predicted_label": 0, "positive_probability": 0.10},
        ]

        summary = evaluate.summarize_scored_rows(scored, precision_top_k=2)

        self.assertEqual(summary["tp"], 1)
        self.assertEqual(summary["fp"], 1)
        self.assertEqual(summary["tn"], 1)
        self.assertEqual(summary["fn"], 1)
        self.assertEqual(summary["positive_samples"], 2)
        self.assertEqual(summary["precision"], 0.5)
        self.assertEqual(summary["positive_recall"], 0.5)
        self.assertEqual(summary["precision@2"], 0.5)
        self.assertEqual(summary["precision@50"], 0.5)

    def test_checkpoint_evaluation_reports_precision_at_50(self):
        class FakeModel:
            training = True

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

        class FakeTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return "prompt"

        rows = [
            {"metadata": {"scode": "000001", "anchor_date": "2026-01-01", "label": 1}},
            {"metadata": {"scode": "000002", "anchor_date": "2026-01-02", "label": 0}},
        ]
        predictions = [
            {"label": "positive", "positive_probability": 0.9},
            {"label": "negative", "positive_probability": 0.1},
        ]
        with TemporaryDirectory() as temp_dir, patch.object(
            train,
            "compact_messages_from_sample",
            return_value=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "positive"}],
        ), patch.object(train, "score_prediction", side_effect=predictions):
            result = train._evaluate_training_checkpoint(
                model=FakeModel(),
                tokenizer=FakeTokenizer(),
                rows=rows,
                output_dir=Path(temp_dir),
                update=100,
                total_updates=1000,
                progress=0.1,
                trained_epochs=0.1,
                threshold=0.5,
                threshold_position=0.2,
                max_samples=0,
                max_seq_length=3072,
                precision_top_k=10,
                precision_threshold=0.4,
            )

        self.assertEqual(result["precision@50"], 0.5)

    def test_checkpoint_eval_path_prefers_external_evaluation_dir(self):
        data_dir = Path("cycle-02") / "datasets" / "training"
        eval_dir = Path("cycle-02") / "datasets" / "evaluation"

        self.assertEqual(train._checkpoint_eval_path(data_dir, eval_dir), eval_dir / "test.jsonl")
        self.assertEqual(train._checkpoint_eval_path(data_dir, None), data_dir / "test.jsonl")

    def test_precision_target_tag_uses_top_k_and_threshold(self):
        self.assertEqual(common.precision_target_tag(20, 0.30), "top20_precision030")
        self.assertEqual(common.precision_target_tag(5, 30), "top5_precision030")
        self.assertEqual(common.normalize_precision_threshold(30), 0.30)

    def test_checkpoint_evaluation_samples_fixed_rows_by_default(self):
        rows = [{"metadata": {"scode": f"{idx:06d}"}} for idx in range(20)]

        sampled_a, method_a, seed_a = train._sample_eval_rows(rows, 5, update=100)
        sampled_b, method_b, seed_b = train._sample_eval_rows(rows, 5, update=100)
        sampled_c, method_c, seed_c = train._sample_eval_rows(rows, 5, update=101)
        all_rows, method_all, seed_all = train._sample_eval_rows(rows, 0, update=100)

        self.assertEqual(method_a, "fixed")
        self.assertEqual(method_b, "fixed")
        self.assertEqual(method_c, "fixed")
        self.assertIsNone(seed_a)
        self.assertIsNone(seed_b)
        self.assertIsNone(seed_c)
        self.assertEqual(sampled_a, sampled_b)
        self.assertEqual(sampled_a, rows[:5])
        self.assertEqual(sampled_a, sampled_c)
        self.assertEqual(all_rows, rows)
        self.assertEqual(method_all, "all")
        self.assertIsNone(seed_all)

    def test_checkpoint_evaluation_can_still_sample_random_rows_explicitly(self):
        rows = [{"metadata": {"scode": f"{idx:06d}"}} for idx in range(20)]

        sampled_a, method_a, seed_a = train._sample_eval_rows(rows, 5, update=100, method="random")
        sampled_b, method_b, seed_b = train._sample_eval_rows(rows, 5, update=101, method="random")

        self.assertEqual(method_a, "random")
        self.assertEqual(method_b, "random")
        self.assertNotEqual(seed_a, seed_b)
        self.assertNotEqual(sampled_a, sampled_b)

    def test_positive_samples_use_twenty_trading_day_cooldown(self):
        trade_dates = [date(2026, 1, day) for day in range(1, 32)]
        events = [
            SampleEvent("000001", date(2026, 1, 1), 1, "positive", 1.0),
            SampleEvent("000001", date(2026, 1, 10), 1, "positive", 2.0),
            SampleEvent("000001", date(2026, 1, 22), 1, "positive", 3.0),
            SampleEvent("000002", date(2026, 1, 10), 1, "positive", 4.0),
        ]

        with patch.object(
            base_build_dataset,
            "load_trading_dates_for_symbols_batched",
            return_value={"000001": trade_dates, "000002": trade_dates},
        ):
            kept = base_build_dataset.apply_positive_cooldown(None, events, 20)

        self.assertEqual([(event.scode, event.anchor_date) for event in kept], [
            ("000001", date(2026, 1, 1)),
            ("000001", date(2026, 1, 22)),
            ("000002", date(2026, 1, 10)),
        ])
        self.assertEqual(base_build_dataset.NEGATIVE_EXCLUSION_TRADING_DAYS, 20)

    def test_evaluation_defaults_match_2026_holdout_period(self):
        with without_project_env():
            args = build_validation_dataset.build_parser().parse_args([])

        self.assertEqual(args.start_date, "20260101")
        self.assertEqual(args.end_date, "20260430")
        self.assertEqual(args.output_dir, common.default_validation_dir("long"))
        self.assertIn("no_partial_week_recall60_long", str(args.output_dir))

    def test_compact_window_and_sequence_length_defaults_match_2048_format(self):
        self.assertEqual(common.COMPACT_DAILY_WINDOW, 13)
        self.assertEqual(common.COMPACT_WEEKLY_WINDOW, 8)
        self.assertEqual(common.COMPACT_MONTHLY_WINDOW, 5)
        self.assertEqual(common.sample_mode_config("short")["daily"], 8)
        self.assertEqual(common.sample_mode_config("short")["weekly"], 5)
        self.assertEqual(common.default_max_seq_length("short"), 1024)
        self.assertIn("data_no_partial_week_recall60_short", str(common.default_data_dir("short")))
        self.assertIn("data_no_partial_week_recall60_long", str(common.default_data_dir("long")))
        self.assertIn("data_no_partial_week_recall60_xlong", str(common.default_data_dir("xlong")))
        self.assertIn("data_no_partial_week_recall60_xxlong", str(common.default_data_dir("xxlong")))
        self.assertIn("data_evaluation_no_partial_week_recall60_short", str(common.default_validation_dir("short")))
        self.assertEqual(common.sample_mode_config("xlong"), {"daily": 21, "weekly": 13, "monthly": 8, "max_seq_length": 3072})
        self.assertEqual(common.default_max_seq_length("xlong"), 3072)
        self.assertEqual(common.sample_mode_config("xxlong"), {"daily": 34, "weekly": 21, "monthly": 13, "max_seq_length": 4096})
        self.assertEqual(common.default_max_seq_length("xxlong"), 4096)
        self.assertEqual(train.build_parser().parse_args([]).max_seq_length, 2048)
        self.assertEqual(evaluate.build_parser().parse_args([]).max_seq_length, 2048)
        self.assertEqual(predict_day.build_parser().parse_args(["--date", "20260514"]).max_seq_length, 2048)

    def test_pick_weekly_window_uses_only_completed_weekly_rows_for_midweek_anchor(self):
        weekly_rows = [
            daily("20251205", 1, 1, 1, 10),
            daily("20251212", 1, 1, 1, 11),
            daily("20251219", 1, 1, 1, 12),
            daily("20251226", 1, 1, 1, 13),
            daily("20260102", 1, 1, 1, 14),
        ]
        daily_rows = [
            daily("20251229", 14, 15, 13, 14),
            daily("20251230", 14, 16, 13, 15),
            daily("20251231", 15, 17, 14, 16),
        ]

        window = common.pick_weekly_window(weekly_rows, daily_rows, date(2025, 12, 31), 4)

        self.assertIsNotNone(window)
        self.assertEqual([row["date"] for row in window], ["20251205", "20251212", "20251219", "20251226"])
        self.assertEqual(window[-1]["close"], 13)

    def test_compact_prompt_uses_short_csv_recent_thirteen_daily_eight_weekly_and_five_monthly(self):
        daily_rows = [daily(f"202601{day:02d}", day, day + 1, day - 1, day + 0.5) for day in range(1, 26)]
        weekly_rows = [daily(f"2025{week:02d}01", week, week + 1, week - 1, week + 0.5) for week in range(1, 16)]
        monthly_rows = [daily(f"2025{month:02d}28", month, month + 1, month - 1, month + 0.5) for month in range(1, 8)]

        prompt = common.build_compact_prompt("000001", date(2026, 1, 10), daily_rows, weekly_rows, monthly_rows)

        self.assertIn("cols=dt/o/h/l/c/v/a/m5/m13/m34/m55", prompt)
        self.assertEqual(prompt.splitlines().count("D"), 1)
        self.assertEqual(prompt.splitlines().count("W"), 1)
        self.assertEqual(prompt.splitlines().count("M"), 1)
        daily_section = prompt.split("D\n", 1)[1].split("\nW\n", 1)[0].splitlines()
        weekly_section = prompt.split("\nW\n", 1)[1].split("\nM\n", 1)[0].splitlines()
        monthly_section = prompt.split("\nM\n", 1)[1].splitlines()
        self.assertEqual(len(daily_section), 13)
        self.assertEqual(len(weekly_section), 8)
        self.assertEqual(len(monthly_section), 5)
        self.assertTrue(daily_section[0].startswith("1,"))
        self.assertTrue(daily_section[-1].startswith("13,"))
        self.assertTrue(weekly_section[0].startswith("1,"))
        self.assertTrue(weekly_section[-1].startswith("8,"))
        self.assertTrue(monthly_section[-1].startswith("5,"))
        self.assertNotIn("260101", prompt)
        self.assertNotIn("260102", prompt)
        self.assertNotIn("260103", prompt)
        self.assertNotIn("260104", prompt)
        self.assertNotIn("260110", prompt)
        self.assertNotIn("250101", prompt)
        self.assertNotIn("250201", prompt)
        self.assertNotIn("250301", prompt)
        self.assertNotIn("250901", prompt)

    def test_compact_prompt_normalizes_prices_volume_and_amount_by_window_average(self):
        daily_rows = [
            daily("20260101", 1, 2, 1, 2, volume=100, amount=1000),
            daily("20260102", 2, 3, 1, 2, volume=200, amount=3000),
        ]
        weekly_rows = [
            daily("20260102", 1, 2, 1, 2, volume=10, amount=50),
            daily("20260109", 2, 3, 1, 2, volume=30, amount=150),
        ]

        prompt = common.build_compact_prompt("000001", date(2026, 1, 10), daily_rows, weekly_rows, daily_window=2, weekly_window=2, monthly_window=0)

        self.assertIn("D\n1,0.5,1,0.5,1,0.67,0.5", prompt)
        self.assertIn("2,1,1.5,0.5,1,1.33,1.5", prompt)
        self.assertIn("W\n1,0.5,1,0.5,1,0.5,0.5", prompt)
        self.assertIn("2,1,1.5,0.5,1,1.5,1.5", prompt)

    def test_sample_modes_require_weekly_ma13_or_monthly_rows(self):
        weekly_without_ma13 = [daily(f"202512{day:02d}", 1, 1, 1, 1) for day in range(1, 6)]
        weekly_with_ma13 = [dict(row, ma13=1.0) for row in weekly_without_ma13]
        monthly_rows = [daily(f"2025{month:02d}28", 1, 1, 1, 1) for month in range(1, 6)]
        anchor_daily = [daily("20260101", 1, 1, 1, 1)]

        self.assertFalse(common._sample_windows_are_valid("short", weekly_without_ma13, [], anchor_daily))
        self.assertTrue(common._sample_windows_are_valid("short", weekly_with_ma13, [], anchor_daily))
        self.assertFalse(common._sample_windows_are_valid("long", weekly_with_ma13, monthly_rows[:4], anchor_daily))
        self.assertTrue(common._sample_windows_are_valid("long", weekly_with_ma13, monthly_rows, anchor_daily))

    def test_sample_modes_require_anchor_close_in_configured_bottom_band(self):
        weekly_rows = [dict(daily(f"202512{day:02d}", 10, 100, 10, 50), ma13=1.0) for day in range(1, 22)]
        monthly_rows = [daily(f"2025{month:02d}28", 10, 100, 10, 50) for month in range(1, 14)]
        low_daily = [daily("20260101", 18, 19, 17, 19)]
        high_daily = [daily("20260101", 30, 31, 29, 30)]

        with patch.dict("os.environ", {"SAMPLE_BOTTOM_BAND_RATIO": "0.10"}):
            self.assertTrue(common._sample_windows_are_valid("short", weekly_rows, [], low_daily))
            self.assertFalse(common._sample_windows_are_valid("short", weekly_rows, [], high_daily))
            self.assertTrue(common._sample_windows_are_valid("long", weekly_rows, monthly_rows[:5], low_daily))
            self.assertFalse(common._sample_windows_are_valid("long", weekly_rows, monthly_rows[:5], high_daily))
            self.assertTrue(common._sample_windows_are_valid("xlong", weekly_rows, monthly_rows[:8], low_daily))
            self.assertTrue(common._sample_windows_are_valid("xxlong", weekly_rows, monthly_rows, low_daily))
            self.assertFalse(common._sample_windows_are_valid("xlong", weekly_rows, monthly_rows[:8], high_daily))
            self.assertFalse(common._sample_windows_are_valid("xxlong", weekly_rows, monthly_rows, high_daily))
            self.assertFalse(common._sample_windows_are_valid("xlong", weekly_rows, monthly_rows[:7], low_daily))

    def test_bottom_band_ratio_can_come_from_environment(self):
        rows = [daily("20251231", 10, 100, 10, 50)]
        anchor_daily = [daily("20260101", 54, 55, 53, 55)]

        with patch.dict("os.environ", {"SAMPLE_BOTTOM_BAND_RATIO": "0.50"}):
            self.assertTrue(common._is_close_in_bottom_band(anchor_daily, rows))
        with patch.dict("os.environ", {"SAMPLE_BOTTOM_BAND_RATIO": "0.10"}):
            self.assertFalse(common._is_close_in_bottom_band(anchor_daily, rows))

    def test_delisted_stock_name_filter(self):
        self.assertTrue(common._looks_delisted_stock_name("\u9000\u5e02\u6d77\u6da6"))
        self.assertTrue(common._looks_delisted_stock_name("\u6d77\u6da6\u9000"))
        self.assertTrue(common._looks_delisted_stock_name("PT\u6c34\u4ed9"))
        self.assertFalse(common._looks_delisted_stock_name("\u5e73\u5b89\u94f6\u884c"))

    def test_abnormal_symbol_filter_marks_delisted_and_long_suspended(self):
        class FakeCursor:
            def __init__(self, responses):
                self.responses = responses

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def execute(self, *_args, **_kwargs):
                return None

            def fetchall(self):
                return self.responses.pop(0)

        class FakeConn:
            def __init__(self, responses):
                self.responses = responses

            def cursor(self):
                return FakeCursor(self.responses)

        recent_dates = [(date(2026, 5, day),) for day in range(20, 9, -1)]
        conn = FakeConn(
            [
                recent_dates,
                [
                    ("000001", "\u5e73\u5b89\u94f6\u884c"),
                    ("000002", "\u9000\u5e02\u6d77\u6da6"),
                    ("000003", "\u6b63\u5e38\u80a1"),
                ],
                [
                    ("000001", date(2026, 5, 20)),
                    ("000002", date(2026, 5, 20)),
                    ("000003", date(2026, 4, 20)),
                ],
            ]
        )

        abnormal = common.load_abnormal_symbols(conn, ["000001", "000002", "000003"], date(2026, 5, 20))

        self.assertEqual(abnormal, {"000002", "000003"})

if __name__ == "__main__":
    unittest.main()
