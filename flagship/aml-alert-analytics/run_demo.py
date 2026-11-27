"""Run the AML alert analytics pipeline end to end.

    python run_demo.py

Generates the synthetic alert feed, ETLs it into a SQLite warehouse
with quarantine, runs the SQL analytics, trains the risk model, and
prints a grounded briefing after verifying every number in it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analytics import load_warehouse, run_analytics
from src.briefing import generate_briefing, verify_briefing
from src.etl import run_etl
from src.generate import generate_raw
from src.model import train_model

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "data" / "warehouse.db"
RAW_PATH = BASE / "data" / "alerts_raw.csv"
SUMMARY_PATH = BASE / "data" / "summary.json"


def main() -> dict:
    print("=" * 68)
    print("AML ALERT ANALYTICS - end-to-end demo")
    print("=" * 68)

    raw = generate_raw()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False)
    print(f"\n[1/5] Generated raw alert feed: {len(raw)} rows "
          f"(saved to data/alerts_raw.csv)")

    etl_counts = run_etl(raw, DB_PATH)
    print(f"[2/5] ETL complete: {etl_counts['quarantined_rows']} quarantined, "
          f"{etl_counts['warehouse_rows']} loaded into SQLite warehouse")

    analytics = run_analytics(DB_PATH)
    print(f"[3/5] SQL analytics: overall confirmed-suspicious rate "
          f"{analytics['overall_rate'] * 100:.1f}% "
          f"({analytics['total_suspicious']} of "
          f"{analytics['total_alerts']} alerts)")
    riskiest = analytics["riskiest_rule"]
    print(f"      Riskiest rule: {riskiest['rule']} "
          f"at {riskiest['rate'] * 100:.1f}%")
    print("      Confirmed-suspicious rate by rule:")
    for rule, block in sorted(
            analytics["by_rule"].items(),
            key=lambda item: item[1]["rate"], reverse=True):
        print(f"        {rule:<20} {block['rate'] * 100:5.1f}%  "
              f"({block['suspicious']}/{block['alerts']})")
    print("      Confirmed-suspicious rate by risk tier:")
    for tier in ("low", "medium", "high"):
        block = analytics["by_risk_tier"][tier]
        print(f"        {tier:<20} {block['rate'] * 100:5.1f}%  "
              f"({block['suspicious']}/{block['alerts']})")
    print("      Confirmed-suspicious rate by amount band:")
    for band in ("under_1k", "1k_to_10k", "10k_to_50k", "50k_plus"):
        block = analytics["by_amount_band"][band]
        print(f"        {band:<20} {block['rate'] * 100:5.1f}%  "
              f"({block['suspicious']}/{block['alerts']})")

    warehouse = load_warehouse(DB_PATH)
    model_metrics = train_model(warehouse)
    print(f"[4/5] Model: test ROC-AUC {model_metrics['roc_auc']:.3f} | "
          f"top-decile capture "
          f"{model_metrics['top_decile_capture_rate'] * 100:.1f}% of "
          f"confirmed-suspicious cases")
    print("      Highest-risk test alerts, with reasons:")
    for example in model_metrics["highest_risk_examples"][:3]:
        reasons = "; ".join(example["reasons"]) or "no single dominant factor"
        print(f"        {example['alert_id']} "
              f"({example['alert_rule']}): risk "
              f"{example['probability']:.3f} - {reasons}")

    briefing = generate_briefing(analytics, model_metrics, etl_counts)
    ungrounded = verify_briefing(briefing, analytics, model_metrics, etl_counts)
    print(f"[5/5] Grounded briefing: {len(briefing['claims'])} claims, "
          f"{ungrounded} ungrounded")
    print("\nBRIEFING")
    print("-" * 68)
    for claim in briefing["claims"]:
        print(f"  - {claim}")
    print("-" * 68)
    assert ungrounded == 0, "Briefing contains ungrounded numbers - do not share it."

    summary = {
        "etl": etl_counts,
        "analytics": analytics,
        "model": model_metrics,
        "briefing_claims": len(briefing["claims"]),
        "briefing_ungrounded": ungrounded,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFull summary written to data/summary.json")
    return summary


if __name__ == "__main__":
    main()
