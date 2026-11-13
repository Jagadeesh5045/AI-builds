"""Run the full flagship end to end and print a JSON summary plus briefing.

Usage: python run_demo.py [--db outputs/fraud_warehouse.db]
Re-running reproduces the exact numbers in the README (seed 42 throughout).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_signal.analytics import run_analytics
from fraud_signal.briefing import build_briefing, top_alert_explanations, verify_briefing
from fraud_signal.generate import generate_raw
from fraud_signal.model import train_and_evaluate
from fraud_signal.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="outputs/fraud_warehouse.db")
    args = parser.parse_args()

    raw = generate_raw()
    pipeline = run_pipeline(raw, args.db)
    analytics = run_analytics(args.db)
    model = train_and_evaluate(pipeline["clean"])
    briefing, claims = build_briefing(
        {"clean_rows": pipeline["clean_rows"], "quarantined_rows": pipeline["quarantined_rows"]},
        analytics,
        model,
    )
    verification = verify_briefing(briefing, claims)
    alerts = top_alert_explanations(model["scored_test"], pipeline["clean"])

    summary = {
        "pipeline": {k: v for k, v in pipeline.items() if k not in ("clean", "quarantine")},
        "headline": analytics["headline"],
        "by_channel": analytics["by_channel"].to_dict("records"),
        "by_category_top3": analytics["by_category"].head(3).to_dict("records"),
        "model": {k: v for k, v in model.items() if k not in ("scored_test", "model")},
        "briefing_verification": verification,
        "top_alerts": alerts,
    }
    out_dir = Path(args.db).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "briefing.txt").write_text(briefing)
    print(json.dumps(summary, indent=2, default=str))
    print("\n--- BRIEFING ---\n")
    print(briefing)


if __name__ == "__main__":
    main()
