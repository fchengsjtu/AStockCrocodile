from __future__ import annotations

from datetime import date

import pandas as pd

PREDICTION_TABLE = "blackbox_predictions"


def ensure_prediction_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} (
                Id BIGINT NOT NULL AUTO_INCREMENT,
                TradeDate DATE NOT NULL,
                StrategyName VARCHAR(64) NOT NULL,
                RankNo INT NOT NULL,
                SCode VARCHAR(10) NOT NULL,
                PositiveProbability DECIMAL(18,10) NOT NULL,
                PositiveLoss DECIMAL(18,10) NULL,
                NegativeLoss DECIMAL(18,10) NULL,
                Threshold DECIMAL(10,6) NOT NULL,
                MaxSeqLength INT NOT NULL,
                CreatedOn DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UpdatedOn DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (Id),
                UNIQUE KEY ux_blackbox_prediction (TradeDate, StrategyName, RankNo),
                UNIQUE KEY ux_blackbox_prediction_code (TradeDate, StrategyName, SCode),
                KEY idx_blackbox_prediction_strategy_date (StrategyName, TradeDate)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
    conn.commit()


def save_top_predictions(
    conn,
    predictions: pd.DataFrame,
    strategy_name: str,
    threshold: float,
    max_seq_length: int,
    top_n: int = 5,
) -> int:
    ensure_prediction_table(conn)
    if predictions.empty or top_n <= 0:
        return 0
    frame = predictions.sort_values("PositiveProbability", ascending=False).head(top_n).copy()
    rows = []
    for rank_no, row in enumerate(frame.itertuples(index=False), start=1):
        rows.append(
            (
                row.TradeDate,
                strategy_name,
                rank_no,
                str(row.SCode),
                float(row.PositiveProbability),
                float(row.PositiveLoss),
                float(row.NegativeLoss),
                float(threshold),
                int(max_seq_length),
            )
        )
    with conn.cursor() as cur:
        trade_date = rows[0][0]
        cur.execute(f"DELETE FROM {PREDICTION_TABLE} WHERE TradeDate = %s AND StrategyName = %s", (trade_date, strategy_name))
        cur.executemany(
            f"""
            INSERT INTO {PREDICTION_TABLE}
                (TradeDate, StrategyName, RankNo, SCode, PositiveProbability, PositiveLoss, NegativeLoss, Threshold, MaxSeqLength)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def first_prediction_date(conn, strategy_name: str) -> date | None:
    ensure_prediction_table(conn)
    with conn.cursor() as cur:
        cur.execute(f"SELECT MIN(TradeDate) FROM {PREDICTION_TABLE} WHERE StrategyName = %s", (strategy_name,))
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def latest_prediction_date(conn, strategy_name: str) -> date | None:
    ensure_prediction_table(conn)
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(TradeDate) FROM {PREDICTION_TABLE} WHERE StrategyName = %s", (strategy_name,))
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def load_prediction_signals(conn, strategy_name: str, start_date: date, end_date: date) -> pd.DataFrame:
    ensure_prediction_table(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                p.TradeDate,
                p.SCode,
                s.SName,
                p.PositiveProbability AS Score,
                CONCAT(
                    p.StrategyName,
                    ': rank=', p.RankNo,
                    '; probability=', p.PositiveProbability,
                    '; positive_loss=', IFNULL(p.PositiveLoss, 0),
                    '; negative_loss=', IFNULL(p.NegativeLoss, 0)
                ) AS Reason,
                p.StrategyName
            FROM {PREDICTION_TABLE} p
            LEFT JOIN stockinfo s ON s.SCode = p.SCode
            WHERE p.StrategyName = %s
              AND p.TradeDate >= %s
              AND p.TradeDate <= %s
            ORDER BY p.TradeDate, p.RankNo
            """,
            (strategy_name, start_date, end_date),
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["TradeDate", "SCode", "SName", "Score", "Reason", "StrategyName"])
