"""Tests for saas-kpi-board. Run with: python -m unittest discover -s tests -v
(also collectable by pytest)."""
from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saas_kpi import db, generate, kpis  # noqa: E402


def _conn(seed=42):
    conn = sqlite3.connect(":memory:")
    db.load(conn, generate.generate(seed=seed))
    return conn


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        a = generate.generate(seed=7)
        b = generate.generate(seed=7)
        self.assertEqual(len(a.customers), len(b.customers))
        self.assertEqual(len(a.snapshots), len(b.snapshots))
        self.assertEqual(a.snapshots[:10], b.snapshots[:10])

    def test_months_cover_18_months(self):
        self.assertEqual(len(generate.MONTHS), 18)
        self.assertEqual(generate.MONTHS[0], "2025-01")
        self.assertEqual(generate.MONTHS[-1], "2026-06")

    def test_churn_month_is_after_signup(self):
        ds = generate.generate()
        for c in ds.customers:
            if c.churn_month is not None:
                self.assertGreater(c.churn_month, c.signup_month)

    def test_no_snapshot_after_churn(self):
        ds = generate.generate()
        churn = {c.customer_id: c.churn_month for c in ds.customers}
        for s in ds.snapshots:
            cm = churn[s["customer_id"]]
            if cm is not None:
                self.assertLess(s["month"], cm)

    def test_snapshot_mrr_matches_plan_price(self):
        ds = generate.generate()
        for s in ds.snapshots:
            self.assertEqual(s["mrr"], generate.PLANS[s["plan"]])


class TestKpis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = _conn()

    def test_monthly_rows_for_every_month(self):
        rows = kpis.monthly_kpis(self.conn)
        self.assertEqual(len(rows), 18)

    def test_mrr_equals_sum_of_snapshots(self):
        rows = kpis.monthly_kpis(self.conn)
        for r in rows:
            total = self.conn.execute(
                "SELECT SUM(mrr) FROM subscription_snapshots WHERE month=?",
                (r["month"],),
            ).fetchone()[0]
            self.assertAlmostEqual(r["mrr"], total, places=2)
            self.assertAlmostEqual(r["arr_run_rate"], r["mrr"] * 12, places=2)

    def test_arpu_consistent(self):
        for r in kpis.monthly_kpis(self.conn):
            self.assertAlmostEqual(r["arpu"], r["mrr"] / r["active_customers"], places=2)

    def test_churn_rates_in_range(self):
        for r in kpis.monthly_kpis(self.conn):
            if r["logo_churn_rate"] is not None:
                self.assertGreaterEqual(r["logo_churn_rate"], 0.0)
                self.assertLess(r["logo_churn_rate"], 0.25)
            if r["revenue_churn_rate"] is not None:
                self.assertGreaterEqual(r["revenue_churn_rate"], 0.0)
                self.assertLess(r["revenue_churn_rate"], 0.25)

    def test_cohort_retention_never_rises_with_age(self):
        rows = kpis.cohort_retention(self.conn)
        by_cohort: dict[str, list] = {}
        for r in rows:
            by_cohort.setdefault(r["cohort"], []).append(r)
        for cohort, lst in by_cohort.items():
            lst.sort(key=lambda x: x["months_since_signup"])
            pcts = [x["retention_pct"] for x in lst]
            self.assertEqual(pcts, sorted(pcts, reverse=True), cohort)
            self.assertAlmostEqual(pcts[0], 100.0, places=1)

    def test_plan_mix_shares_sum_to_100(self):
        rows = kpis.plan_mix(self.conn)
        self.assertAlmostEqual(sum(r["mrr_share_pct"] for r in rows), 100.0, delta=0.3)

    def test_channel_economics_ratios_positive(self):
        for r in kpis.channel_economics(self.conn):
            self.assertGreater(r["avg_cac"], 0)
            self.assertIsNotNone(r["ltv"])
            self.assertGreater(r["ltv"], 0)
            self.assertGreater(r["cac_payback_months"], 0)
            self.assertGreater(r["ltv_cac_ratio"], 0)

    def test_retention_rates_ordering(self):
        ret = kpis.retention_rates(self.conn, lag_months=12)
        self.assertIsNotNone(ret)
        # NRR can exceed GRR because upgrades are uncapped in NRR.
        self.assertGreaterEqual(ret["nrr_pct"], ret["grr_pct"])
        self.assertGreater(ret["grr_pct"], 0)
        self.assertLess(ret["grr_pct"], 100)

    def test_sql_file_in_sync_with_code(self):
        sql_path = Path(__file__).resolve().parents[1] / "sql" / "queries.sql"
        text = sql_path.read_text(encoding="utf-8")
        for name in kpis.SQL:
            self.assertIn(f"-- name: {name}", text)


if __name__ == "__main__":
    unittest.main()
