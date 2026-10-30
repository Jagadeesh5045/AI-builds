"""Smoke tests: generator shape, PSI sanity, registry round-trip."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mlops_credit.data_gen import generate_credit_data
from mlops_credit.psi import psi
from mlops_credit.registry import append_entry, load_registry


def test_data_generator_shape_and_target():
    df = generate_credit_data(n_samples=500, seed=42)
    assert df.shape == (500, 8)
    assert set(df["default"].unique()) <= {0, 1}
    assert df["default"].mean() > 0.05  # a real default signal exists
    assert df["default"].mean() < 0.95
    # Reproducibility: same seed, same frame.
    df2 = generate_credit_data(n_samples=500, seed=42)
    assert df.equals(df2)


def test_psi_identical_distributions_near_zero():
    df = generate_credit_data(n_samples=1000, seed=1)
    value = psi(df["income"], df["income"])
    assert abs(value) < 1e-6


def test_psi_shifted_distribution_is_large():
    df = generate_credit_data(n_samples=1000, seed=1)
    shifted = df["income"] * 0.5
    assert psi(df["income"], shifted) > 0.25


def test_registry_append_and_read_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "registry.json")
        entry = append_entry(path, "LogisticRegression", {"max_iter": 1000},
                             {"roc_auc": 0.8}, "champion")
        assert entry["version"] == "v1"
        assert entry["status"] == "champion"
        entries = load_registry(path)
        assert len(entries) == 1
        assert entries[0]["metrics"]["roc_auc"] == 0.8
        entry2 = append_entry(path, "RandomForestClassifier",
                              {"n_estimators": 200}, {"roc_auc": 0.82},
                              "challenger")
        assert entry2["version"] == "v2"
        assert len(load_registry(path)) == 2
