import sqlite3

import pytest

from fraud_signal.analytics import run_analytics
from fraud_signal.briefing import build_briefing, top_alert_explanations, verify_briefing
from fraud_signal.generate import generate_raw
from fraud_signal.model import train_and_evaluate
from fraud_signal.pipeline import run_pipeline


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    raw = generate_raw()
    db = tmp_path_factory.mktemp("wh") / "fraud.db"
    pipeline = run_pipeline(raw, db)
    return {"raw": raw, "db": db, "pipeline": pipeline}


def test_raw_shape_and_dirty_injection(built):
    # 12,000 clean + 75 injected dirty rows (40 duplicates, 35 defective).
    assert len(built["raw"]) == 12_075


def test_quarantine_counts_exact(built):
    p = built["pipeline"]
    assert p["raw_rows"] == 12_075
    assert p["quarantined_rows"] == 75
    assert p["clean_rows"] == 12_000
    assert p["raw_rows"] == p["clean_rows"] + p["quarantined_rows"]
    by_check = p["quarantine_by_check"]
    assert by_check["txn_id_unique"] == 40
    assert by_check["amount_positive"] == 17  # 12 negative + 5 zero
    assert by_check["customer_id_present"] == 10
    assert by_check["channel_valid"] == 8


def test_warehouse_tables_reconcile(built):
    with sqlite3.connect(built["db"]) as conn:
        txn = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        quar = conn.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0]
    assert txn == 12_000
    assert quar == 75


def test_analytics_headline_consistent(built):
    a = run_analytics(built["db"])
    h = a["headline"]
    assert h["transactions"] == 12_000
    assert 0 < h["fraud_rate_pct"] < 15
    assert h["fraud_value_gbp"] < h["total_value_gbp"]
    assert set(a["by_channel"]["channel"]) == {"in_store", "online", "contactless", "atm"}
    # Fraud rows are worth more on average than legit rows in this scenario.
    split = a["amount_split"].set_index("is_fraud")
    assert split.loc[1, "avg_amount_gbp"] > split.loc[0, "avg_amount_gbp"]


def test_model_quality(built):
    m = train_and_evaluate(built["pipeline"]["clean"])
    assert m["test_rows"] == 2_400
    assert m["roc_auc"] > 0.80
    assert m["top_decile_fraud_capture"] > 0.40
    assert len(m["top_features"]) == 5


def test_briefing_is_fully_grounded(built):
    a = run_analytics(built["db"])
    m = train_and_evaluate(built["pipeline"]["clean"])
    briefing, claims = build_briefing(
        {"clean_rows": 12_000, "quarantined_rows": 75}, a, m
    )
    result = verify_briefing(briefing, claims)
    assert result["ungrounded_count"] == 0
    assert result["claims_checked"] == len(claims) == 12


def test_alert_explanations_give_reasons(built):
    m = train_and_evaluate(built["pipeline"]["clean"])
    alerts = top_alert_explanations(m["scored_test"], built["pipeline"]["clean"])
    assert len(alerts) == 5
    for alert in alerts:
        assert alert["fraud_probability"] >= 0.5
        assert alert["reasons"]
