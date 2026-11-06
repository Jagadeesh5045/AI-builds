import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_wait.analytics import deprivation_gap, overall_metrics
from care_wait.briefing import generate_briefing, verify_briefing
from care_wait.data_gen import generate_clean, generate_raw
from care_wait.model import train_breach_model
from care_wait.quality import run_quality_checks


def test_clean_generation_is_deterministic():
    a = generate_clean(n=200)
    b = generate_clean(n=200)
    assert a.equals(b)
    assert len(a) == 200


def test_quality_layer_quarantines_dirty_rows():
    raw = generate_raw()
    clean, quarantine, counts = run_quality_checks(raw)
    assert len(raw) == 6075
    assert len(clean) == 6000
    assert len(quarantine) == 75
    assert counts["duplicate_referral_id"] == 30
    assert counts["missing_specialty"] == 20
    assert counts["negative_wait"] == 15
    assert counts["invalid_age_band"] == 10


def test_analytics_are_consistent():
    clean, _, _ = run_quality_checks(generate_raw())
    overall = overall_metrics(clean)
    assert overall["n_referrals"] == 6000
    assert overall["breach_n"] == int((clean["wait_days"] > 126).sum())
    gap = deprivation_gap(clean)
    assert gap["median_gap_days"] == gap["median_wait_q5"] - gap["median_wait_q1"]


def test_breach_model_beats_chance():
    clean, _, _ = run_quality_checks(generate_raw())
    metrics = train_breach_model(clean)
    assert metrics["roc_auc"] >= 0.65
    assert metrics["n_train"] + metrics["n_test"] == 6000
    assert metrics["top_feature"] in metrics["feature_importance_grouped"]


def test_briefing_is_fully_grounded():
    from care_wait.analytics import specialty_medians

    clean, _, _ = run_quality_checks(generate_raw())
    overall = overall_metrics(clean)
    medians = specialty_medians(clean)
    gap = deprivation_gap(clean)
    model_metrics = train_breach_model(clean)
    text = generate_briefing(overall, medians, gap, model_metrics)
    assert verify_briefing(text, overall, medians, gap, model_metrics) == []
    tampered = text + " The backlog fell by 999 places."
    assert "999" in verify_briefing(tampered, overall, medians, gap, model_metrics)
