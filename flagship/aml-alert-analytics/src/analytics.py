"""SQL analytics over the AML alert warehouse.

All headline numbers come from SQL queries against SQLite, grouped
rates, counts and medians, not from ad-hoc dataframe arithmetic, so
the same queries could be pointed at a real warehouse unchanged.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _rate_block(rows) -> dict:
    out = {}
    for label, total, suspicious in rows:
        out[str(label)] = {
            "alerts": int(total),
            "suspicious": int(suspicious),
            "rate": round(suspicious / total, 4) if total else 0.0,
        }
    return out


def _median_amount(conn: sqlite3.Connection, suspicious_flag: int) -> float:
    count = conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE confirmed_suspicious = ?", (suspicious_flag,)
    ).fetchone()[0]
    if count == 0:
        return 0.0
    middle = conn.execute(
        "SELECT transaction_amount FROM alerts WHERE confirmed_suspicious = ? "
        "ORDER BY transaction_amount LIMIT 2 OFFSET ?",
        (suspicious_flag, (count - 1) // 2),
    ).fetchall()
    if count % 2 == 1:
        return float(middle[0][0])
    return round((middle[0][0] + middle[1][0]) / 2.0, 2)


def run_analytics(db_path: str | Path) -> dict:
    """Run the full analytics pack and return it as a structured dict."""
    conn = sqlite3.connect(db_path)
    try:
        total, suspicious = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(confirmed_suspicious), 0) FROM alerts"
        ).fetchone()

        by_rule = _rate_block(conn.execute(
            "SELECT alert_rule, COUNT(*), SUM(confirmed_suspicious) "
            "FROM alerts GROUP BY alert_rule ORDER BY alert_rule"
        ).fetchall())

        by_channel = _rate_block(conn.execute(
            "SELECT channel, COUNT(*), SUM(confirmed_suspicious) "
            "FROM alerts GROUP BY channel ORDER BY channel"
        ).fetchall())

        by_tier = _rate_block(conn.execute(
            "SELECT customer_risk_tier, COUNT(*), SUM(confirmed_suspicious) "
            "FROM alerts GROUP BY customer_risk_tier ORDER BY customer_risk_tier"
        ).fetchall())

        by_amount_band = _rate_block(conn.execute(
            "SELECT CASE WHEN transaction_amount < 1000 THEN 'under_1k' "
            "WHEN transaction_amount < 10000 THEN '1k_to_10k' "
            "WHEN transaction_amount < 50000 THEN '10k_to_50k' "
            "ELSE '50k_plus' END AS band, "
            "COUNT(*), SUM(confirmed_suspicious) FROM alerts GROUP BY band"
        ).fetchall())

        by_disposition = _rate_block(conn.execute(
            "SELECT disposition, COUNT(*), SUM(confirmed_suspicious) "
            "FROM alerts GROUP BY disposition ORDER BY disposition"
        ).fetchall())

        median_suspicious = _median_amount(conn, 1)
        median_not = _median_amount(conn, 0)
    finally:
        conn.close()

    riskiest_rule = max(by_rule.items(), key=lambda item: item[1]["rate"])
    riskiest_channel = max(by_channel.items(), key=lambda item: item[1]["rate"])
    return {
        "total_alerts": int(total),
        "total_suspicious": int(suspicious),
        "overall_rate": round(suspicious / total, 4) if total else 0.0,
        "by_rule": by_rule,
        "riskiest_rule": {
            "rule": riskiest_rule[0],
            "rate": riskiest_rule[1]["rate"],
            "alerts": riskiest_rule[1]["alerts"],
            "suspicious": riskiest_rule[1]["suspicious"],
        },
        "by_channel": by_channel,
        "riskiest_channel": {
            "channel": riskiest_channel[0],
            "rate": riskiest_channel[1]["rate"],
            "alerts": riskiest_channel[1]["alerts"],
            "suspicious": riskiest_channel[1]["suspicious"],
        },
        "by_risk_tier": by_tier,
        "by_amount_band": by_amount_band,
        "by_disposition": by_disposition,
        "median_amount_suspicious": median_suspicious,
        "median_amount_not_suspicious": median_not,
    }


def load_warehouse(db_path: str | Path):
    """Load the clean warehouse table as a DataFrame for modelling."""
    import pandas as pd

    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM alerts", conn)
    finally:
        conn.close()
