import unittest
from datetime import date, datetime

import pandas as pd

import a_share_crawler as crawler


class ExRightsTests(unittest.TestCase):
    def test_parse_tencent_exrights_content(self):
        bonus, transfer, cash = crawler.parse_tencent_exrights_content("10\u90016\u80a1, 10\u8f6c2\u80a1, 10\u6d3e1.7\u5143")

        self.assertEqual(bonus, 6.0)
        self.assertEqual(transfer, 2.0)
        self.assertEqual(cash, 1.7)

    def test_normalize_exrights_records_maps_tencent_fields(self):
        records = [
            {
                "SECURITY_CODE": "000001",
                "nd": "2012",
                "fh_sh": "1.315",
                "djr": "2013-06-19",
                "cqr": "2013-06-20",
                "FHcontent": "10\u90016\u80a1, 10\u6d3e1.7\u5143",
            }
        ]

        df = crawler.normalize_exrights_records(records, symbol="000001")

        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["SCode"], "000001")
        self.assertEqual(row["ReportDate"], date(2012, 12, 31))
        self.assertEqual(row["EquityRecordDate"], date(2013, 6, 19))
        self.assertEqual(row["ExDividendDate"], date(2013, 6, 20))
        self.assertEqual(row["NoticeDate"], date(2013, 6, 20))
        self.assertEqual(row["AssignProgress"], "Tencent exrights")
        self.assertEqual(row["ImplPlanProfile"], "10\u90016\u80a1, 10\u6d3e1.7\u5143")
        self.assertEqual(row["BonusRatio"], 6.0)
        self.assertTrue(pd.isna(row["TransferRatio"]))
        self.assertEqual(row["BonusItRatio"], 6.0)
        self.assertEqual(row["PretaxBonusRmb"], 1.7)

    def test_normalize_exrights_records_uses_symbol_when_code_missing(self):
        records = [
            {
                "nd": "2024",
                "djr": "2025-04-01",
                "cqr": "2025-04-02",
                "FHcontent": "10\u8f6c2\u80a1",
            }
        ]

        df = crawler.normalize_exrights_records(records, symbol="600000")

        self.assertEqual(df.iloc[0]["SCode"], "600000")
        self.assertEqual(df.iloc[0]["ReportDate"], date(2024, 12, 31))
        self.assertEqual(df.iloc[0]["TransferRatio"], 2.0)

    def test_rows_for_exrights_insert_shapes_values(self):
        df = pd.DataFrame(
            [
                {
                    "SCode": "000001",
                    "SName": None,
                    "ReportDate": date(2025, 12, 31),
                    "PlanNoticeDate": None,
                    "EquityRecordDate": date(2026, 6, 1),
                    "ExDividendDate": date(2026, 6, 2),
                    "NoticeDate": date(2026, 6, 2),
                    "AssignProgress": "Tencent exrights",
                    "ImplPlanProfile": "10\u6d3e3.60\u5143",
                    "BonusItRatio": pd.NA,
                    "BonusRatio": pd.NA,
                    "TransferRatio": pd.NA,
                    "PretaxBonusRmb": 3.6,
                    "DividendRatio": pd.NA,
                    "BasicEps": pd.NA,
                    "Bvps": pd.NA,
                    "PerCapitalReserve": pd.NA,
                    "PerUnassignProfit": pd.NA,
                    "PnpYoyRatio": pd.NA,
                    "TotalShares": pd.NA,
                }
            ]
        )

        rows = crawler.rows_for_exrights_insert(df)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), 24)
        self.assertEqual(rows[0][0], "000001|2025-12-31|2026-06-02|2026-06-02")
        self.assertEqual(len(rows[0][1]), 64)
        self.assertEqual(rows[0][2], "000001")
        self.assertIsNone(rows[0][11])
        self.assertEqual(rows[0][14], 3.6)
        self.assertIsInstance(rows[0][22], datetime)

    def test_exrights_content_hash_changes_when_material_field_changes(self):
        base = pd.DataFrame(
            [
                {
                    "SCode": "000001",
                    "SName": None,
                    "ReportDate": date(2025, 12, 31),
                    "PlanNoticeDate": None,
                    "EquityRecordDate": date(2026, 6, 1),
                    "ExDividendDate": date(2026, 6, 2),
                    "NoticeDate": date(2026, 6, 2),
                    "AssignProgress": "Tencent exrights",
                    "ImplPlanProfile": "10\u6d3e3.60\u5143",
                    "BonusItRatio": pd.NA,
                    "BonusRatio": pd.NA,
                    "TransferRatio": pd.NA,
                    "PretaxBonusRmb": 3.6,
                    "DividendRatio": pd.NA,
                    "BasicEps": pd.NA,
                    "Bvps": pd.NA,
                    "PerCapitalReserve": pd.NA,
                    "PerUnassignProfit": pd.NA,
                    "PnpYoyRatio": pd.NA,
                    "TotalShares": pd.NA,
                }
            ]
        )
        changed = base.copy()
        changed.loc[0, "PretaxBonusRmb"] = 4.0

        base_hash = crawler.rows_for_exrights_insert(base)[0][1]
        changed_hash = crawler.rows_for_exrights_insert(changed)[0][1]

        self.assertNotEqual(base_hash, changed_hash)

    def test_parser_accepts_exrights_command(self):
        args = crawler.build_parser().parse_args(["exrights", "--workers", "4", "--truncate", "--no-refresh-klines"])

        self.assertEqual(args.command, "exrights")
        self.assertEqual(args.workers, 4)
        self.assertTrue(args.truncate)
        self.assertTrue(args.no_refresh_klines)


    def test_stockinfo_upsert_does_not_write_contenthash(self):
        class FakeCursor:
            def __init__(self):
                self.sql = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def execute(self, sql, params=None):
                self.sql.append(sql)
                if "SHOW COLUMNS FROM stockinfo" in sql:
                    self.fetchone_result = None

            def fetchone(self):
                return getattr(self, "fetchone_result", None)

            def executemany(self, sql, rows):
                self.sql.append(sql)

        class FakeConn:
            def __init__(self):
                self.cursor_obj = FakeCursor()

            def cursor(self):
                return self.cursor_obj

            def commit(self):
                pass

        conn = FakeConn()
        stocks = pd.DataFrame([{"code": "000001", "name": "Ping An Bank"}])

        crawler.upsert_stock_info(conn, stocks)

        stockinfo_sql = "\n".join(conn.cursor_obj.sql)
        self.assertIn("INSERT INTO stockinfo", stockinfo_sql)
        self.assertNotIn("ContentHash = VALUES(ContentHash)", stockinfo_sql)

    def test_stockinfo_contenthash_default_is_fixed_when_column_exists(self):
        class FakeCursor:
            def __init__(self):
                self.sql = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def execute(self, sql, params=None):
                self.sql.append(sql)

            def fetchone(self):
                return ("ContentHash", "char(64)", "NO", "", None, "")

        class FakeConn:
            def __init__(self):
                self.cursor_obj = FakeCursor()
                self.commits = 0

            def cursor(self):
                return self.cursor_obj

            def commit(self):
                self.commits += 1

        conn = FakeConn()

        crawler.ensure_stockinfo_contenthash_default(conn)

        sql = "\n".join(conn.cursor_obj.sql)
        self.assertIn("UPDATE stockinfo SET ContentHash = ''", sql)
        self.assertIn("ALTER TABLE stockinfo MODIFY COLUMN ContentHash CHAR(64) NOT NULL DEFAULT ''", sql)
        self.assertEqual(conn.commits, 1)

    def test_run_parser_defaults_to_qfq_only(self):
        args = crawler.build_parser().parse_args(["run"])

        self.assertEqual(args.adjust, "qfq")


if __name__ == "__main__":
    unittest.main()
