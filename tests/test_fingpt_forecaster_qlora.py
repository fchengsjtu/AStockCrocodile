import tempfile
import unittest
from pathlib import Path

from fingpt_forecaster_qlora import common
from fingpt_forecaster_qlora import evaluate
from fingpt_forecaster_qlora import train_qlora
from fingpt_forecaster_qlora.common import load_key_value_file


class FinGptForecasterQloraTests(unittest.TestCase):
    def test_load_key_value_file_accepts_powershell_env_assignments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "env.txt"
            path.write_text(
                "$env:MYSQL_HOST='127.0.0.1'\n"
                "$env:MYSQL_USER='fcheng'; $env:MYSQL_PASSWORD='123456'\n"
                "MYSQL_DATABASE=emstocks\n",
                encoding="utf-8",
            )

            values = load_key_value_file(path)

        self.assertEqual(values["MYSQL_HOST"], "127.0.0.1")
        self.assertEqual(values["MYSQL_USER"], "fcheng")
        self.assertEqual(values["MYSQL_PASSWORD"], "123456")
        self.assertEqual(values["MYSQL_DATABASE"], "emstocks")

    def test_resolve_mysql_host_keeps_localhost_outside_wsl(self):
        original = common.is_wsl
        common.is_wsl = lambda: False
        try:
            self.assertEqual(common.resolve_mysql_host("127.0.0.1"), "127.0.0.1")
        finally:
            common.is_wsl = original

    def test_resolve_mysql_host_uses_wsl_windows_host_for_localhost(self):
        original_is_wsl = common.is_wsl
        original_detect = common.detect_wsl_windows_host
        common.is_wsl = lambda: True
        common.detect_wsl_windows_host = lambda: "172.28.64.1"
        try:
            self.assertEqual(common.resolve_mysql_host("127.0.0.1"), "172.28.64.1")
            self.assertEqual(common.resolve_mysql_host("localhost"), "172.28.64.1")
            self.assertEqual(common.resolve_mysql_host("192.168.1.10"), "192.168.1.10")
        finally:
            common.is_wsl = original_is_wsl
            common.detect_wsl_windows_host = original_detect

    def test_mysql_host_not_allowed_message_includes_grant_sql(self):
        message = common.mysql_host_not_allowed_message(
            {"MYSQL_USER": "fcheng", "MYSQL_DATABASE": "emstocks"},
            "172.20.160.1",
            "Host '172.20.167.134' is not allowed to connect to this MySQL server",
        )

        self.assertIn("'fcheng'@'172.20.167.134'", message)
        self.assertIn("GRANT ALL PRIVILEGES ON emstocks.*", message)
        self.assertIn("'fcheng'@'172.%'", message)
        self.assertIn("172.20.160.1", message)

    def test_mysql_auth_dependency_message_mentions_cryptography(self):
        message = common.mysql_auth_dependency_message(
            "'cryptography' package is required for sha256_password or caching_sha2_password auth methods"
        )

        self.assertIn("cryptography", message)
        self.assertIn("python -m pip install", message)
        self.assertIn("requirements.txt", message)

    def test_model_load_error_mentions_offline_options(self):
        error = train_qlora.model_load_error("NousResearch/Llama-2-7b-chat-hf", OSError("network unreachable"))
        message = str(error)

        self.assertIn("HF_ENDPOINT", message)
        self.assertIn("BASE_MODEL", message)
        self.assertIn("dataset-only", message)
        self.assertIn("NO_FORECASTER_ADAPTER", message)

    def test_missing_dataset_error_mentions_generation_command(self):
        error = train_qlora.missing_dataset_error(
            Path("fingpt_forecaster_qlora/data/smoke"),
            Path("fingpt_forecaster_qlora/data/smoke/train.jsonl"),
            Path("fingpt_forecaster_qlora/data/smoke/valid.jsonl"),
        )
        message = str(error)

        self.assertIn("dataset-only", message)
        self.assertIn("build_dataset", message)
        self.assertIn("fingpt_forecaster_qlora", message)
        self.assertIn("data", message)
        self.assertIn("smoke", message)

    def test_missing_adapter_error_mentions_training_command(self):
        error = evaluate.missing_adapter_error(Path("fingpt_forecaster_qlora/runs/smoke-qwen-0.5b/adapter"))
        message = str(error)

        self.assertIn("adapter_config.json", message)
        self.assertIn("train_qlora", message)
        self.assertIn("smoke-qwen-0.5b", message)
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", message)

    def test_normalize_training_argument_keys_accepts_eval_strategy_rename(self):
        kwargs = train_qlora.normalize_training_argument_keys(
            {"evaluation_strategy": "steps", "output_dir": "tmp"},
            {"eval_strategy", "output_dir"},
        )

        self.assertEqual(kwargs["eval_strategy"], "steps")
        self.assertNotIn("evaluation_strategy", kwargs)


if __name__ == "__main__":
    unittest.main()
