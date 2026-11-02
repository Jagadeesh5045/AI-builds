"""Entry point: generate data, load SQLite, run the SQL KPI suite, write outputs.

Usage:
    python main.py [--seed 42] [--out output]

Outputs (in --out):
    saas.db               SQLite warehouse
    report.md             analyst report with headline KPIs and notes
    monthly_kpis.csv      one row per month
    cohort_retention.csv  one row per cohort x month
    channel_economics.csv one row per acquisition channel
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from saas_kpi import db, generate, kpis, report  # noqa: E402


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run(seed: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate.generate(seed=seed)
    db_path = out_dir / "saas.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        db.load(conn, dataset)
        monthly = kpis.monthly_kpis(conn)
        cohorts = kpis.cohort_retention(conn)
        plans = kpis.plan_mix(conn)
        channels = kpis.channel_economics(conn)
        retention = kpis.retention_rates(conn, lag_months=12)
        failed = kpis._rows(conn, "failed_payments")
    finally:
        conn.close()

    (out_dir / "report.md").write_text(
        report.build_report(monthly, cohorts, plans, channels, retention, failed),
        encoding="utf-8",
    )
    _write_csv(out_dir / "monthly_kpis.csv", monthly)
    _write_csv(out_dir / "cohort_retention.csv", cohorts)
    _write_csv(out_dir / "channel_economics.csv", channels)

    latest = monthly[-1]
    return {
        "customers_total": len(dataset.customers),
        "snapshots": len(dataset.snapshots),
        "months": len(monthly),
        "latest_month": latest["month"],
        "latest_mrr": latest["mrr"],
        "latest_active_customers": latest["active_customers"],
        "latest_arpu": latest["arpu"],
        "retention_12m": retention,
        "best_channel_ltv_cac": max(channels, key=lambda r: (r["ltv_cac_ratio"] or 0))["channel"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SaaS KPI board")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("output"))
    args = ap.parse_args()
    summary = run(args.seed, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
