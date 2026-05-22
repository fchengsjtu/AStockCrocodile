import importlib
import unittest
from datetime import date
from unittest.mock import patch


RANDOM_NEGATIVE_TARGETS = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]


class FakeCursor:
    def __init__(self):
        self.sql = ""
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self.sql = sql
        self.params = params

    def fetchall(self):
        return [
            ("000001", date(2026, 1, 2)),
            ("000001", date(2026, 1, 3)),
            ("000002", date(2026, 1, 6)),
            ("000002", date(2026, 1, 6)),
        ]


class FakeConnection:
    def __init__(self):
        self.last_cursor = None

    def cursor(self):
        self.last_cursor = FakeCursor()
        return self.last_cursor


class BlackboxRandomNegativeTests(unittest.TestCase):
    def test_target_builders_use_random_negative_loader(self):
        for target in RANDOM_NEGATIVE_TARGETS:
            with self.subTest(target=target):
                module = importlib.import_module(f"blackbox_finetune_recall{target}.build_dataset")

                self.assertTrue(hasattr(module, "load_random_negative_events"))
                self.assertFalse(hasattr(module, "load_negative_events"))

    def test_random_negative_loader_uses_rand_and_filters_excluded_dates(self):
        module = importlib.import_module("blackbox_finetune_recall80.build_dataset")
        conn = FakeConnection()

        with (
            patch.object(module, "load_excluded_positive_windows", return_value={"000001": {date(2026, 1, 2)}}),
            patch.object(module, "excluded_dates_by_symbol", return_value={"000001": {date(2026, 1, 2)}}),
        ):
            events = module.load_random_negative_events(
                conn,
                "short_term_surge_3d_20pct",
                date(2026, 1, 1),
                date(2026, 1, 31),
                2,
                20260518,
                80,
            )

        self.assertIn("ORDER BY RAND(%s)", conn.last_cursor.sql)
        self.assertEqual(conn.last_cursor.params[2], 20260518)
        self.assertEqual([(event.scode, event.anchor_date, event.label) for event in events], [
            ("000001", date(2026, 1, 3), 0),
            ("000002", date(2026, 1, 6), 0),
        ])


if __name__ == "__main__":
    unittest.main()
