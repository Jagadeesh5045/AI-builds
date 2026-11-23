import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from calibration import IsotonicCalibrator, PlattScaler
from data import generate_readmissions, stratified_split
from metrics import brier, expected_calibration_error, roc_auc
from model import LogisticModel
from pipeline import run
from predict import score


def test_data_deterministic_and_shaped():
    X1, y1, f1 = generate_readmissions(n=200, seed=3)
    X2, y2, _ = generate_readmissions(n=200, seed=3)
    assert X1.shape == (200, 7) and len(f1) == 7
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert 0.05 < y1.mean() < 0.6


def test_split_is_stratified_and_complete():
    _, y, _ = generate_readmissions(n=500, seed=1)
    tr, va, te = stratified_split(y)
    assert len(tr) + len(va) + len(te) == 500
    for part in (tr, va, te):
        assert 0.05 < y[part].mean() < 0.6


def test_auc_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert abs(roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9


def test_ece_zero_for_perfect_probabilities():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert expected_calibration_error(y, p) == 0.0
    assert brier(y, p) == 0.0


def test_model_learns_signal():
    X, y, _ = generate_readmissions(n=1200, seed=5)
    tr, va, _ = stratified_split(y)
    m = LogisticModel(epochs=500).fit(X[tr], y[tr])
    assert roc_auc(y[va], m.predict_proba(X[va])) > 0.70


def test_isotonic_output_is_monotonic():
    rng = np.random.default_rng(0)
    p = np.sort(rng.random(300))
    y = (rng.random(300) < p).astype(int)
    cal = IsotonicCalibrator().fit(p, y)
    out = cal.predict(np.linspace(0, 1, 50))
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0 and out.max() <= 1


def test_platt_returns_valid_probabilities():
    rng = np.random.default_rng(1)
    p = rng.random(300)
    y = (rng.random(300) < p).astype(int)
    out = PlattScaler().fit(p, y).predict(p)
    assert out.min() > 0 and out.max() < 1


def test_pipeline_end_to_end(tmp_path):
    res = run(out_dir=str(tmp_path), n=900, seed=11)
    assert res["chosen_calibration"] in ("raw", "platt", "isotonic")
    assert 0.05 <= res["threshold"]["chosen"] <= 0.95
    assert res["test_calibrated"]["roc_auc"] > 0.65
    card = json.loads((tmp_path / "model_card.json").read_text())
    s = score(card, {"age": 80, "prior_admissions": 3})
    assert 0 <= s["readmission_risk"] <= 1
    assert isinstance(s["flag_for_follow_up"], bool)
