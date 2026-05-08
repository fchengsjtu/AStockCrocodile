import unittest
from datetime import date, datetime

import pandas as pd

import a_share_crawler as crawler


class ExRightsTests(unittest.TestCase):
    def test_normalize_exrights_records_maps_eastmoney_fields(self):
        records = [
            {
                "SECURITY_CODE": "000001",
                "SECURITY_NAME_ABBR": "平安银行",
                "REPORT_DATE": "2025-12-31 00:00:00",
                "PLAN_NOTICE_DATE": "2026-03-21 00:00:00",
                "EQUITY_RECORD_DATE": "2026-06-01 00:00:00",
                "EX_DIVIDEND_DATE": "2026-06-02 00:00:00",
                "NOTICE_DATE": "2026-03-21 00:00:00",
                "ASSIGN_PROGRESS": "董事会决议通过",
                "IMPL_PLAN_PROFILE": "10派3.60元(含税)",
                "BONUS_IT_RATIO": 1.5,
                "BONUS_RATIO": 0.5,
                "IT_RATIO": 1.0,
                "PRETAX_BONUS_RMB": 3.6,
                "DIVIDENT_RATIO": 0.0316,
                "BASIC_EPS": 2.07,
                "BVPS": 23.25,
                "PER_CAPITAL_RESERVE": 4.15,
                "PER_UNASSIGN_PROFIT": 13.95,
                "PNP_YOY_RATIO": -4.2,
                "TOTAL_SHARES": 19405918198,
            }
        ]

        df = crawler.normalize_exrights_records(records)

        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["SCode"], "000001")
        self.assertEqual(row["SName"], "平安银行")
        self.assertEqual(row["ReportDate"], date(2025, 12, 31))
        self.assertEqual(row["ExDividendDate"], date(2026, 6, 2))
        self.assertEqual(row["ImplPlanProfile"], "10派3.60元(含税)")
        self.assertAlmostEqual(row["PretaxBonusRmb"], 3.6)
        self.assertAlmostEqual(row["TransferRatio"], 1.0)

    def test_normalize_exrights_records_uses_symbol_when_code_missing(self):
        records = [
            {
                "SECURITY_NAME_ABBR": "测试股票",
                "REPORT_DATE": "2024-12-31",
                "PUBLISH_DATE": "2025-04-01",
            }
        ]

        df = crawler.normalize_exrights_records(records, symbol="600000")

        self.assertEqual(df.iloc[0]["SCode"], "600000")
        self.assertEqual(df.iloc[0]["NoticeDate"], date(2025, 4, 1))

    def test_rows_for_exrights_insert_shapes_values(self):
        df = pd.DataFrame(
            [
                {
                    "SCode": "000001",
                    "SName": "平安银行",
                    "ReportDate": date(2025, 12, 31),
                    "PlanNoticeDate": date(2026, 3, 21),
                    "EquityRecordDate": None,
                    "ExDividendDate": None,
                    "NoticeDate": date(2026, 3, 21),
                    "AssignProgress": "预案",
                    "ImplPlanProfile": "10派3.60元(含税)",
                    "BonusItRatio": pd.NA,
                    "BonusRatio": pd.NA,
                    "TransferRatio": pd.NA,
                    "PretaxBonusRmb": 3.6,
                    "DividendRatio": 0.0316,
                    "BasicEps": 2.07,
                    "Bvps": 23.25,
                    "PerCapitalReserve": 4.15,
                    "PerUnassignProfit": 13.95,
                    "PnpYoyRatio": -4.2,
                    "TotalShares": 19405918198,
                }
            ]
        )

        rows = crawler.rows_for_exrights_insert(df)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), 23)
        self.assertEqual(rows[0][0], "000001|2025-12-31||2026-03-21")
        self.assertEqual(rows[0][1], "000001")
        self.assertIsNone(rows[0][10])
        self.assertEqual(rows[0][13], 3.6)
        self.assertIsInstance(rows[0][21], datetime)

    def test_parser_accepts_exrights_command(self):
        args = crawler.build_parser().parse_args(["exrights", "--workers", "4", "--truncate"])

        self.assertEqual(args.command, "exrights")
        self.assertEqual(args.workers, 4)
        self.assertTrue(args.truncate)


if __name__ == "__main__":
    unittest.main()
