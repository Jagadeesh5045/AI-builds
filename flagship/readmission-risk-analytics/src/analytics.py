"""SQL analytics over the discharge warehouse.

All headline numbers come from SQL queries against SQLite - grouped
rates, counts and medians - not from ad-hoc dataframe arithmetic, so the
same queries could be pointed at a real warehouse unchanged.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _rate_block(rows) -> dict:
    """Shape ``(label, discharges, readmissions)`` rows into a dict."""
    out = {}
    for label, total, readmitted in rows:
        out[str(label)] = {
            "discharges": int(total),
            "readmissions": int(readmitted),
            "rate": round(readmitted / total, 4) if total else 0.0,
        }
    return out


def _median(conn: sqlite3.Connection, readmitted_flag: int) -> float:
    """Median length of stay for one group, fetched with SQL paging."""
    count = conn.execute(
        "SELECT COUNT(*) FROM discharges WHERE readmitted_30d = ?", (readmitted_flag,)
    ).fetchone()[0]
    if count == 0:
        return 0.0
    middle = conn.execute(
        "SELECT length_of_stay_days FROM discharges WHERE readmitted_30d = ? "
        "ORDER BY length_of_stay_days LIMIT 2 OFFSET ?",
        (readmitted_flag, (count - 1) // 2),
    ).fetchall()
    if count % 2 == 1:
        return float(middle[0][0])
    return (middle[0][0] + middle[1][0]) / 2.0


def run_analytics(db_path: str | Path) -> dict:
    """Run the full analytics pack and return it as a structured dict."""
    conn = sqlite3.connect(db_path)
    try:
        total, readmitted = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(readmitted_30d), 0) FROM discharges"
        ).fetchone()

        by_condition = _rate_block(conn.execute(
            "SELECT primary_condition, COUNT(*), SUM(readmitted_30d) "
            "FROM discharges GROUP BY primary_condition ORDER BY primary_condition"
        ).fetchall())

        by_prior_band = _rate_block(conn.execute(
            "SELECT CASE WHEN prior_admissions = 0 THEN '0' "
            "WHEN prior_admissions = 1 THEN '1' "
            "WHEN prior_admissions = 2 THEN '2' ELSE '3+' END AS band, "
            "COUNT(*), SUM(readmitted_30d) FROM discharges "
            "GROUP BY band ORDER BY band"
        ).fetchall())

        by_destination = _rate_block(conn.execute(
            "SELECT discharge_destination, COUNT(*), SUM(readmitted_30d) "
            "FROM discharges GROUP BY discharge_destination "
            "ORDER BY discharge_destination"
        ).fetchall())

        by_deprivation = _rate_block(conn.execute(
            "SELECT deprivation_quintile, COUNT(*), SUM(readmitted_30d) "
            "FROM discharges GROUP BY deprivation_quintile "
            "ORDER BY deprivation_quintile"
        ).fetchall())

        by_admission = _rate_block(conn.execute(
            "SELECT admission_type, COUNT(*), SUM(readmitted_30d) "
            "FROM discharges GROUP BY admission_type ORDER BY admission_type"
        ).fetchall())

        median_readmitted = _median(conn, 1)
        median_not = _median(conn, 0)
    finally:
        conn.close()

    riskiest = max(by_condition.items(), key=lambda item: item[1]["rate"])
    return {
        "total_discharges": int(total),
        "total_readmissions": int(readmitted),
        "overall_rate": round(readmitted / total, 4) if total else 0.0,
        "by_condition": by_condition,
        "riskiest_condition": {
            "condition": riskiest[0],
            "rate": riskiest[1]["rate"],
            "discharges": riskiest[1]["discharges"],
            "readmissions": riskiest[1]["readmissions"],
        },
        "by_prior_admissions_band": by_prior_band,
        "by_discharge_destination": by_destination,
        "by_deprivation_quintile": by_deprivation,
        "by_admission_type": by_admission,
        "median_los_readmitted": median_readmitted,
        "median_los_not_readmitted": median_not,
    }


def load_warehouse(db_path: str | Path):
    """Load the clean warehouse table as a DataFrame for modelling."""
    import pandas as pd

    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM discharges", conn)
    finally:
        conn.close()
