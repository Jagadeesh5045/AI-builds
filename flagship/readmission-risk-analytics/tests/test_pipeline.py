"""End-to-end tests for the readmission risk pipeline.

The pipeline is run once (module scope) against a temporary warehouse,
then each stage is checked against the contract it promises: the ETL
quarantines exactly the damaged records, the analytics describe the
warehouse that was actually loaded, the model beats a sensible quality
bar, and the briefing contains no ungrounded numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics import load_warehouse, run_analytics
from src.briefing import generate_briefing, verify_briefing
from src.etl import run_etl
from src.generate import CLEAN_ROWS, CONDITIONS, CORRUPT_ROWS, generate_raw
from src.model import risk_reasons, train_model


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("warehouse") / "warehouse.db"
    raw = generate_raw()
    etl_counts = run_etl(raw, db_path)
    analytics = run_analytics(db_path)
    model_metrics = train_model(load_warehouse(db_path))
    briefing = generate_briefing(analytics, model_metrics, etl_counts)
    return {
        "raw_rows": len(raw),
        "etl": etl_counts,
        "analytics": analytics,
        "model": model_metrics,
        "briefing": briefing,
    }


def test_raw_feed_size(pipeline):
    assert pipeline["raw_rows"] == CLEAN_ROWS + CORRUPT_ROWS


def test_etl_quarantine_count(pipeline):
    assert pipeline["etl"]["quarantined_rows"] == CORRUPT_ROWS
    # Every quarantined record carries a reason.
    assert pipeline["etl"]["quarantine_reasons"]
    assert sum(pipeline["etl"]["quarantine_reasons"].values()) == CORRUPT_ROWS


def test_warehouse_row_count(pipeline):
    assert pipeline["etl"]["warehouse_rows"] == CLEAN_ROWS
    assert (pipeline["etl"]["raw_rows"]
            == pipeline["etl"]["warehouse_rows"]
            + pipeline["etl"]["quarantined_rows"])


def test_analytics_sanity(pipeline):
    analytics = pipeline["analytics"]
    assert analytics["total_discharges"] == CLEAN_ROWS
    assert analytics["total_readmissions"] == sum(
        block["readmissions"] for block in analytics["by_condition"].values()
    )
    assert 0.05 < analytics["overall_rate"] < 0.30
    assert set(analytics["by_condition"]) == set(CONDITIONS)
    assert analytics["riskiest_condition"]["condition"] in CONDITIONS
    assert analytics["riskiest_condition"]["rate"] >= analytics["overall_rate"]
    assert set(analytics["by_prior_admissions_band"]) == {"0", "1", "2", "3+"}
    # The risk structure built into the data should show through:
    # patients with 3+ prior admissions readmit more than first-timers.
    bands = analytics["by_prior_admissions_band"]
    assert bands["3+"]["rate"] > bands["0"]["rate"]
    assert analytics["median_los_readmitted"] >= analytics["median_los_not_readmitted"]


def test_model_quality(pipeline):
    model_metrics = pipeline["model"]
    assert model_metrics["roc_auc"] > 0.65
    # The top decile should capture far more than a random tenth would.
    assert model_metrics["top_decile_capture_rate"] > 0.20
    assert model_metrics["top_decile_patients"] > 0
    assert model_metrics["highest_risk_examples"]


def test_briefing_is_grounded(pipeline):
    briefing = pipeline["briefing"]
    assert 8 <= len(briefing["claims"]) <= 12
    ungrounded = verify_briefing(
        briefing, pipeline["analytics"], pipeline["model"], pipeline["etl"]
    )
    assert ungrounded == 0


def test_briefing_verifier_catches_invention(pipeline):
    briefing = {"claims": ["Exactly 9999 patients were readmitted."]}
    ungrounded = verify_briefing(
        briefing, pipeline["analytics"], pipeline["model"], pipeline["etl"]
    )
    assert ungrounded == 1


def test_risk_reasons_are_plain_english():
    row = {
        "prior_admissions": 3,
        "comorbidity_count": 5,
        "admission_type": "emergency",
        "length_of_stay_days": 14,
        "primary_condition": "heart failure",
        "age_band": "85+",
        "discharge_destination": "care home",
    }
    reasons = risk_reasons(row)
    assert "3 prior admissions in the last year" in reasons
    assert "emergency admission" in reasons
    assert len(reasons) >= 4
