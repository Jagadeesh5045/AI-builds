"""Analytics layer. Every published number comes from these functions,
run as SQL against the warehouse where a query is the natural fit."""
import sqlite3

import pandas as pd


def _query(db_path, sql):
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def overall_metrics(df):
    return {
        "n_referrals": int(len(df)),
        "median_wait_days": float(df["wait_days"].median()),
        "mean_wait_days": round(float(df["wait_days"].mean()), 1),
        "breach_n": int(df["breach_18wk"].sum()),
        "breach_pct": round(100.0 * float(df["breach_18wk"].mean()), 1),
    }


def by_specialty(db_path):
    return _query(
        db_path,
        """
        SELECT specialty,
               COUNT(*) AS n,
               AVG(wait_days) AS mean_wait,
               SUM(breach_18wk) AS breach_n,
               ROUND(100.0 * AVG(breach_18wk), 1) AS breach_pct
        FROM referrals GROUP BY specialty ORDER BY mean_wait DESC
        """,
    )


def specialty_medians(df):
    out = df.groupby("specialty")["wait_days"].median().sort_values(ascending=False)
    return {str(k): float(v) for k, v in out.items()}


def by_month(df):
    d = df.copy()
    d["referral_month"] = pd.to_datetime(d["referral_date"]).dt.strftime("%Y-%m")
    g = d.groupby("referral_month").agg(
        n=("wait_days", "size"),
        median_wait=("wait_days", "median"),
        breach_pct=("breach_18wk", lambda s: round(100.0 * s.mean(), 1)),
    )
    return g.reset_index()


def by_priority(df):
    g = df.groupby("priority").agg(
        n=("wait_days", "size"),
        median_wait=("wait_days", "median"),
        breach_pct=("breach_18wk", lambda s: round(100.0 * s.mean(), 1)),
    )
    return g.reset_index()


def deprivation_gap(df):
    q1 = df[df["imd_quintile"] == 1]
    q5 = df[df["imd_quintile"] == 5]
    return {
        "median_wait_q1": float(q1["wait_days"].median()),
        "median_wait_q5": float(q5["wait_days"].median()),
        "median_gap_days": float(q5["wait_days"].median() - q1["wait_days"].median()),
        "breach_pct_q1": round(100.0 * float(q1["breach_18wk"].mean()), 1),
        "breach_pct_q5": round(100.0 * float(q5["breach_18wk"].mean()), 1),
    }
