import tempfile
import unittest
from pathlib import Path

from fingpt_forecaster_qlora import common
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


if __name__ == "__main__":
    unittest.main()
