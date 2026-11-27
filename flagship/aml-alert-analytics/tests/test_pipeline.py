"""End-to-end tests for the AML alert analytics pipeline.

The pipeline is run once (module scope) against a temporary warehouse,
then each stage is checked against the contract it promises: the ETL
quarantines exactly the damaged records, the analytics describe the
warehouse that was actually loaded, the model beats a sensible
quality bar, and the briefing contains no ungrounded numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics import load_warehouse, run_analytics
from src.briefing import generate_briefing, verify_briefing
from src.etl import run_etl
from src.generate import ALERT_RULES, CHANNELS, CLEAN_ROWS, CORRUPT_ROWS, generate_raw
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
    assert pipeline["etl"]["quarantine_reasons"]
    assert sum(pipeline["etl"]["quarantine_reasons"].values()) == CORRUPT_ROWS


def test_warehouse_row_count(pipeline):
    assert pipeline["etl"]["warehouse_rows"] == CLEAN_ROWS
    assert (pipeline["etl"]["raw_rows"]
            == pipeline["etl"]["warehouse_rows"]
            + pipeline["etl"]["quarantined_rows"])


def test_analytics_sanity(pipeline):
    analytics = pipeline["analytics"]
    assert analytics["total_alerts"] == CLEAN_ROWS
    assert analytics["total_suspicious"] == sum(
        block["suspicious"] for block in analytics["by_rule"].values()
    )
    assert analytics["total_suspicious"] == sum(
        block["suspicious"] for block in analytics["by_channel"].values()
    )
    assert 0.03 < analytics["overall_rate"] < 0.40
    assert set(analytics["by_rule"]) == set(ALERT_RULES)
    assert set(analytics["by_channel"]) == set(CHANNELS)
    assert set(analytics["by_amount_band"]) == {"under_1k", "1k_to_10k", "10k_to_50k", "50k_plus"}
    assert analytics["riskiest_rule"]["rule"] in ALERT_RULES
    assert analytics["riskiest_rule"]["rate"] >= analytics["overall_rate"]
    tiers = analytics["by_risk_tier"]
    assert tiers["high"]["rate"] > tiers["low"]["rate"]
    bands = analytics["by_amount_band"]
    assert bands["50k_plus"]["rate"] > bands["under_1k"]["rate"]
    assert analytics["by_disposition"]["confirmed_suspicious"]["alerts"] == analytics["total_suspicious"]


def test_model_quality(pipeline):
    model_metrics = pipeline["model"]
    assert model_metrics["roc_auc"] > 0.70
    assert model_metrics["top_decile_capture_rate"] > 0.25
    assert model_metrics["top_decile_alerts"] > 0
    assert model_metrics["highest_risk_examples"]


def test_briefing_is_grounded(pipeline):
    briefing = pipeline["briefing"]
    assert 8 <= len(briefing["claims"]) <= 12
    ungrounded = verify_briefing(
        briefing, pipeline["analytics"], pipeline["model"], pipeline["etl"]
    )
    assert ungrounded == 0


def test_briefing_verifier_catches_invention(pipeline):
    briefing = {"claims": ["Exactly 9999 alerts were confirmed suspicious."]}
    ungrounded = verify_briefing(
        briefing, pipeline["analytics"], pipeline["model"], pipeline["etl"]
    )
    assert ungrounded == 1


def test_risk_reasons_are_plain_english():
    row = {
        "transaction_amount": 75000.00,
        "customer_risk_tier": "high",
        "alert_rule": "structuring",
        "prior_alerts": 3,
        "account_age_days": 90,
        "channel": "wire",
    }
    reasons = risk_reasons(row)
    assert "customer in the high risk tier" in reasons
    assert "3 prior alerts for this customer" in reasons
    assert len(reasons) >= 4
