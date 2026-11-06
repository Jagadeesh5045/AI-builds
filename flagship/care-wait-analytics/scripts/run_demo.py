"""End-to-end demo: generate, check, load, analyse, model, brief.

Run: python scripts/run_demo.py
Writes outputs/report.json and outputs/briefing.md, prints a summary.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_wait.analytics import by_month, by_priority, by_specialty, deprivation_gap, overall_metrics, specialty_medians
from care_wait.briefing import generate_briefing, verify_briefing
from care_wait.data_gen import generate_raw
from care_wait.model import train_breach_model
from care_wait.quality import run_quality_checks
from care_wait.warehouse import load_warehouse, read_warehouse


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    db_path = str(out / "care_wait.db")

    raw = generate_raw()
    clean, quarantine, counts = run_quality_checks(raw)
    loaded = load_warehouse(clean, db_path)
    df = read_warehouse(db_path)

    overall = overall_metrics(df)
    spec = by_specialty(db_path)
    medians = specialty_medians(df)
    months = by_month(df)
    priority = by_priority(df)
    gap = deprivation_gap(df)
    model_metrics = train_breach_model(df)
    briefing = generate_briefing(overall, medians, gap, model_metrics)
    ungrounded = verify_briefing(briefing, overall, medians, gap, model_metrics)

    report = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "quarantined_rows": int(len(quarantine)),
        "quality_failures_by_check": counts,
        "warehouse_rows": loaded,
        "overall": overall,
        "by_specialty": spec.to_dict("records"),
        "specialty_medians": medians,
        "by_month": months.to_dict("records"),
        "by_priority": priority.to_dict("records"),
        "deprivation_gap": gap,
        "model": model_metrics,
        "briefing_ungrounded_numbers": ungrounded,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "briefing.md").write_text(briefing + "\n", encoding="utf-8")

    print(f"raw={len(raw)} clean={len(clean)} quarantined={len(quarantine)} warehouse={loaded}")
    print(f"quality_failures={counts}")
    print(f"overall={overall}")
    print("by_specialty:")
    print(spec.to_string(index=False))
    print("by_month:")
    print(months.to_string(index=False))
    print("by_priority:")
    print(priority.to_string(index=False))
    print(f"deprivation_gap={gap}")
    print(f"model={model_metrics}")
    print(f"briefing_ungrounded_numbers={ungrounded}")
    print("\n" + briefing)


if __name__ == "__main__":
    main()
