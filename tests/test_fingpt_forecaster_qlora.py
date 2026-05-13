import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
