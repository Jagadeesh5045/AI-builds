"""Unit tests for the drift-guard pipeline.

Run:  python -m pytest tests/ -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

import data
import drift as drift_mod
import registry


@pytest.fixture(scope="module")
def df():
    return data.load_dataset()


def test_dataset_validates(df):
    assert len(df) == 569
    assert df["target"].nunique() == 2


def test_split_is_stratified_and_reproducible(df):
    a = data.split_data(df)
    b = data.split_data(df)
    pd.testing.assert_frame_equal(a[0], b[0])
    # positive rate preserved within 2 points on both splits
    base = df["target"].mean()
    assert abs(a[2].mean() - base) < 0.02
    assert abs(a[3].mean() - base) < 0.02


def test_validate_rejects_bad_frames(df):
    with pytest.raises(ValueError):
        data.validate_dataframe(df.drop(columns=["target"]))
    with pytest.raises(ValueError):
        data.validate_dataframe(df.head(0))


def test_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY_FILE", tmp_path / "exp.jsonl")
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    exp_id = registry.log_experiment("dummy", {"a": 1},
                                     {"roc_auc": 0.9, "f1": 0.8},
                                     train_rows=10, feature_names=["x"])
    assert registry.list_experiments()[0]["experiment_id"] == exp_id
    # deterministic id: logging the same config twice yields the same id
    exp_id2 = registry.log_experiment("dummy", {"a": 1},
                                      {"roc_auc": 0.9, "f1": 0.8},
                                      train_rows=10, feature_names=["x"])
    assert exp_id == exp_id2


def test_champion_picks_best_cv_score(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY_FILE", tmp_path / "exp.jsonl")
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    registry.log_experiment("worse", {}, {"roc_auc": 0.5}, 10, ["x"])
    registry.log_experiment("better", {}, {"roc_auc": 0.99}, 10, ["x"])
    assert registry.champion_experiment()["model"] == "better"


def test_psi_zero_for_identical_distributions(df):
    X_train, _, _, _ = data.split_data(df)
    result = drift_mod.check_batch(X_train, X_train.sample(n=100, random_state=0))
    assert result["verdict"] == "OK"
    assert result["max_psi"] < 0.25


def test_psi_flags_shifted_batch(df):
    X_train, X_test, _, _ = data.split_data(df)
    shifted = drift_mod.simulate_shifted_batch(X_test)
    result = drift_mod.check_batch(X_train, shifted)
    assert result["verdict"] == "DRIFT"
    assert len(result["drifted_features"]) >= 2


def test_train_produces_champion_and_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY_FILE", tmp_path / "exp.jsonl")
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    import train as train_mod
    champion_id = train_mod.main()
    champ = registry.champion_experiment()
    assert champ["experiment_id"] == champion_id
    assert (tmp_path / "models" / f"{champion_id}.joblib").exists()
    assert champ["cv_scores"]["roc_auc"] > 0.9  # this dataset is learnable
