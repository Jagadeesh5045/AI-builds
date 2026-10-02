"""End-to-end smoke tests for the ticket-triage pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_gen import CATEGORIES, generate
from src.drift import psi
from src.train import train

TINY_DIR = ROOT / "tests" / "_tiny"


@pytest.fixture(scope="module")
def tiny_project(tmp_path_factory):
    """Train on a tiny 200-sample dataset so the test suite stays fast."""
    base = tmp_path_factory.mktemp("tiny")
    csv_path = base / "tickets.csv"
    models_dir = base / "models"
    rows = generate(n=200, seed=7)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    train(csv_path, models_dir, test_size=0.25, seed=7)
    return base, csv_path, models_dir


def test_data_gen_file(tmp_path):
    out = tmp_path / "tickets.csv"
    rows = generate(n=200, seed=123)
    pd.DataFrame(rows).to_csv(out, index=False)
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == ["id", "text", "category", "priority"]
    assert len(df) == 200
    assert set(df["category"].unique()) == set(CATEGORIES)
    assert df["text"].str.len().min() > 10
    # reproducible: same seed -> same first row
    rows2 = generate(n=200, seed=123)
    assert rows[0]["text"] == rows2[0]["text"]


def test_training_smoke(tiny_project):
    base, csv_path, models_dir = tiny_project
    metrics_path = models_dir / "metrics.json"
    assert metrics_path.exists()
    import json
    metrics = json.loads(metrics_path.read_text())
    # tiny synthetic data should be comfortably learnable
    assert metrics["category"]["accuracy"] > 0.75
    assert metrics["priority"]["accuracy"] > 0.55
    for f in ("vectorizer.joblib", "category_model.joblib",
              "priority_model.joblib", "baseline_distribution.json"):
        assert (models_dir / f).exists()


def test_api_endpoints(tiny_project, monkeypatch):
    base, csv_path, models_dir = tiny_project
    monkeypatch.setattr("api.main.MODELS_DIR", models_dir)
    import api.main as app_module
    from fastapi.testclient import TestClient
    with TestClient(app_module.app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"

        r = client.post("/triage", json={"text": "I was charged twice on my credit card this month, please refund the duplicate"})
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {"category", "confidence", "priority", "suggested_reply"}
        assert body["category"] == "billing"
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["priority"] in ("low", "medium", "high", "urgent")

        r = client.post("/triage/batch", json={"texts": [
            "The app crashes on startup after the update",
            "Where is my parcel, tracking has not moved in days",
        ]})
        assert r.status_code == 200
        assert r.json()["count"] == 2

        r = client.post("/triage", json={"text": "   "})
        assert r.status_code in (400, 422)

        r = client.get("/metrics")
        assert r.status_code == 200
        assert "served_metrics" in r.json() and "drift" in r.json()


def test_drift_psi_sanity():
    base = {"billing": 0.5, "technical": 0.5}
    assert psi(base, base) < 1e-6  # identical distributions -> ~0
    assert psi(base, {"billing": 0.9, "technical": 0.1}) > 0.25  # shifted -> large
