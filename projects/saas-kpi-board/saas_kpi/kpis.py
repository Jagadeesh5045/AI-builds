"""KPI definitions, each one a named SQL query plus a thin Python wrapper.

Definitions used (written down because every team argues about these):
- MRR: sum of snapshot MRR for the month. ARR run rate = MRR x 12.
- ARPU: MRR / active customers that month.
- Logo churn rate: customers active last month and gone this month,
  divided by customers active last month.
- Revenue churn rate: last month's MRR from those churned customers,
  divided by last month's total MRR.
- Cohort retention: for a signup cohort, share still active N months later.
- GRR / NRR at 12 months: for customers active 12 months ago, their MRR
  now (churned = 0) divided by their MRR then. GRR caps each customer's
  current MRR at their starting MRR, NRR does not.
- LTV by channel: average MRR per customer x gross margin / monthly churn
  rate for that channel. Payback months: CAC / (avg MRR x gross margin).
"""
from __future__ import annotations

import sqlite3

from .generate import GROSS_MARGIN

SQL = {
    "monthly_kpis": """
        WITH m AS (
            SELECT month, COUNT(*) AS active_customers, SUM(mrr) AS mrr
            FROM subscription_snapshots GROUP BY month
        ),
        new_c AS (
            SELECT signup_month AS month, COUNT(*) AS new_customers
            FROM customers GROUP BY signup_month
        ),
        churned AS (
            SELECT churn_month AS month, COUNT(*) AS churned_customers
            FROM customers WHERE churn_month IS NOT NULL GROUP BY churn_month
        )
        SELECT m.month, m.active_customers, ROUND(m.mrr, 2) AS mrr,
               ROUND(m.mrr * 12, 2) AS arr_run_rate,
               ROUND(m.mrr / m.active_customers, 2) AS arpu,
               COALESCE(n.new_customers, 0) AS new_customers,
               COALESCE(c.churned_customers, 0) AS churned_customers
        FROM m
        LEFT JOIN new_c n ON n.month = m.month
        LEFT JOIN churned c ON c.month = m.month
        ORDER BY m.month
    """,
    "cohort_retention": """
        SELECT c.signup_month AS cohort, s.month AS month,
               COUNT(DISTINCT s.customer_id) AS retained,
               (SELECT COUNT(*) FROM customers c2 WHERE c2.signup_month = c.signup_month) AS cohort_size,
               ROUND(SUM(s.mrr), 2) AS retained_mrr
        FROM customers c
        JOIN subscription_snapshots s ON s.customer_id = c.customer_id
        GROUP BY c.signup_month, s.month
        ORDER BY c.signup_month, s.month
    """,
    "plan_mix": """
        SELECT month, plan, COUNT(*) AS customers, ROUND(SUM(mrr), 2) AS mrr
        FROM subscription_snapshots
        WHERE month = (SELECT MAX(month) FROM subscription_snapshots)
        GROUP BY month, plan ORDER BY mrr DESC
    """,
    "channel_economics": """
        SELECT c.channel,
               COUNT(DISTINCT c.customer_id) AS customers,
               ROUND(AVG(c.cac), 2) AS avg_cac,
               ROUND(SUM(i.amount), 2) AS revenue_collected,
               ROUND(AVG(s.mrr), 2) AS avg_mrr
        FROM customers c
        JOIN subscription_snapshots s ON s.customer_id = c.customer_id
        JOIN invoices i ON i.customer_id = c.customer_id AND i.month = s.month
        GROUP BY c.channel ORDER BY revenue_collected DESC
    """,
    "failed_payments": """
        SELECT month, COUNT(*) AS invoices, SUM(1 - paid) AS failed,
               ROUND(100.0 * SUM(1 - paid) / COUNT(*), 2) AS failed_pct
        FROM invoices GROUP BY month ORDER BY month
    """,
}


def _rows(conn: sqlite3.Connection, name: str) -> list[dict]:
    cur = conn.execute(SQL[name])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def monthly_kpis(conn: sqlite3.Connection) -> list[dict]:
    rows = _rows(conn, "monthly_kpis")
    prev = None
    for r in rows:
        if prev is None:
            r["logo_churn_rate"] = None
            r["revenue_churn_rate"] = None
            r["mrr_growth_pct"] = None
        else:
            churned = r["churned_customers"]
            r["logo_churn_rate"] = round(churned / prev["active_customers"], 4)
            # MRR lost to this month's churners, valued at last month's prices,
            # is approximated from churned count x last month ARPU when the
            # exact per-customer split is not carried in this summary query.
            lost = _churned_mrr(conn, r["month"])
            r["revenue_churn_rate"] = round(lost / prev["mrr"], 4) if prev["mrr"] else None
            r["mrr_growth_pct"] = round(100.0 * (r["mrr"] - prev["mrr"]) / prev["mrr"], 2)
        prev = r
    return rows


def _churned_mrr(conn: sqlite3.Connection, churn_month: str) -> float:
    """MRR in their final active month of customers who churned into churn_month."""
    row = conn.execute(
        """
        SELECT COALESCE(SUM(s.mrr), 0) FROM subscription_snapshots s
        JOIN customers c ON c.customer_id = s.customer_id
        WHERE c.churn_month = ?
          AND s.month = (SELECT MAX(s2.month) FROM subscription_snapshots s2
                         WHERE s2.customer_id = s.customer_id)
        """,
        (churn_month,),
    ).fetchone()
    return float(row[0])


def cohort_retention(conn: sqlite3.Connection) -> list[dict]:
    rows = _rows(conn, "cohort_retention")
    from .generate import MONTHS
    for r in rows:
        r["months_since_signup"] = MONTHS.index(r["month"]) - MONTHS.index(r["cohort"])
        r["retention_pct"] = round(100.0 * r["retained"] / r["cohort_size"], 1)
    return rows


def plan_mix(conn: sqlite3.Connection) -> list[dict]:
    rows = _rows(conn, "plan_mix")
    total = sum(r["mrr"] for r in rows) or 1.0
    for r in rows:
        r["mrr_share_pct"] = round(100.0 * r["mrr"] / total, 1)
    return rows


def channel_economics(conn: sqlite3.Connection) -> list[dict]:
    rows = _rows(conn, "channel_economics")
    # Monthly churn by channel: churned customers divided by total
    # customer-months observed for that channel. Simple, stated, reproducible.
    monthly_churn = {}
    for channel, churned, snapshots in conn.execute(
        """
        SELECT c.channel, COUNT(DISTINCT c.customer_id),
               (SELECT COUNT(*) FROM subscription_snapshots s
                JOIN customers c2 ON c2.customer_id = s.customer_id
                WHERE c2.channel = c.channel)
        FROM customers c WHERE c.churn_month IS NOT NULL GROUP BY c.channel
        """
    ).fetchall():
        monthly_churn[channel] = (churned / snapshots) if snapshots else 0.0
    for r in rows:
        mc = monthly_churn.get(r["channel"], 0.0)
        r["monthly_churn_rate"] = round(mc, 4)
        margin_mrr = r["avg_mrr"] * GROSS_MARGIN
        r["ltv"] = round(margin_mrr / mc, 2) if mc > 0 else None
        r["cac_payback_months"] = round(r["avg_cac"] / margin_mrr, 2) if margin_mrr else None
        r["ltv_cac_ratio"] = round(r["ltv"] / r["avg_cac"], 2) if r["ltv"] and r["avg_cac"] else None
    return rows


def retention_rates(conn: sqlite3.Connection, lag_months: int = 12) -> dict | None:
    """GRR and NRR at a fixed lag, over customers active in the base month."""
    months = [r[0] for r in conn.execute(
        "SELECT DISTINCT month FROM subscription_snapshots ORDER BY month")]
    if len(months) <= lag_months:
        return None
    base, current = months[-(lag_months + 1)], months[-1]
    rows = conn.execute(
        """
        SELECT b.mrr AS base_mrr, COALESCE(n.mrr, 0) AS current_mrr
        FROM subscription_snapshots b
        LEFT JOIN subscription_snapshots n
          ON n.customer_id = b.customer_id AND n.month = ?
        WHERE b.month = ?
        """,
        (current, base),
    ).fetchall()
    base_total = sum(r[0] for r in rows)
    if not base_total:
        return None
    nrr = sum(r[1] for r in rows) / base_total
    grr = sum(min(r[1], r[0]) for r in rows) / base_total
    return {
        "base_month": base, "current_month": current, "lag_months": lag_months,
        "customers_in_base": len(rows),
        "grr_pct": round(100.0 * grr, 1), "nrr_pct": round(100.0 * nrr, 1),
    }
