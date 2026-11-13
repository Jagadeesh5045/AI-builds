"""SQL analytics over the warehouse. Every headline number in the README,
the briefing and the showcase comes from one of these queries."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def _q(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def run_analytics(db_path: str | Path) -> dict:
    with sqlite3.connect(db_path) as conn:
        headline = _q(
            conn,
            """
            SELECT COUNT(*) AS transactions,
                   ROUND(SUM(amount_gbp), 2) AS total_value_gbp,
                   SUM(is_fraud) AS fraud_transactions,
                   ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct,
                   ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount_gbp ELSE 0 END), 2) AS fraud_value_gbp
            FROM transactions
            """,
        ).iloc[0].to_dict()

        by_channel = _q(
            conn,
            """
            SELECT channel, COUNT(*) AS transactions, SUM(is_fraud) AS fraud_transactions,
                   ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct,
                   ROUND(AVG(amount_gbp), 2) AS avg_amount_gbp
            FROM transactions GROUP BY channel ORDER BY fraud_rate_pct DESC
            """,
        )

        by_category = _q(
            conn,
            """
            SELECT merchant_category, COUNT(*) AS transactions, SUM(is_fraud) AS fraud_transactions,
                   ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct
            FROM transactions GROUP BY merchant_category ORDER BY fraud_rate_pct DESC
            """,
        )

        amount_split = _q(
            conn,
            """
            SELECT is_fraud, ROUND(AVG(amount_gbp), 2) AS avg_amount_gbp,
                   ROUND(MAX(amount_gbp), 2) AS max_amount_gbp, COUNT(*) AS transactions
            FROM transactions GROUP BY is_fraud ORDER BY is_fraud
            """,
        )

        night_split = _q(
            conn,
            """
            SELECT CASE WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 0 AND 5
                        THEN 'night_00_05' ELSE 'day_06_23' END AS period,
                   COUNT(*) AS transactions,
                   ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct
            FROM transactions GROUP BY period ORDER BY period
            """,
        )

    headline["fraud_value_share_pct"] = round(
        100.0 * headline["fraud_value_gbp"] / headline["total_value_gbp"], 2
    )
    int_fields = {"transactions", "fraud_transactions"}
    return {
        "headline": {
            k: (int(v) if k in int_fields else float(v)) for k, v in headline.items()
        },
        "by_channel": by_channel,
        "by_category": by_category,
        "amount_split": amount_split,
        "night_split": night_split,
    }
