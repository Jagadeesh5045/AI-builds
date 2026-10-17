"""Local experiment registry: JSONL experiment log + joblib model artifacts.

Deliberately dependency-free (no MLflow server needed): each training run
appends one JSON record to experiments.jsonl and stores the fitted pipeline
under artifacts/models/<experiment_id>.joblib. The "champion" is the run
with the best cross-validated CHAMPION_METRIC.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from config import MODELS_DIR, REGISTRY_FILE


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_experiment_id(model_name: str, params: dict) -> str:
    """Deterministic id from model name + hyperparams, so re-runs are idempotent."""
    digest = hashlib.sha256(
        json.dumps({"model": model_name, "params": params}, sort_keys=True).encode()
    ).hexdigest()[:10]
    return f"{model_name}-{digest}"


def log_experiment(
    model_name: str,
    params: dict,
    cv_scores: dict[str, float],
    train_rows: int,
    feature_names: list[str],
) -> str:
    """Append one experiment record; return its experiment id."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    experiment_id = make_experiment_id(model_name, params)
    record = {
        "experiment_id": experiment_id,
        "timestamp": _utc_now(),
        "model": model_name,
        "params": params,
        "cv_scores": cv_scores,
        "cv_champion_metric": cv_scores.get("roc_auc"),
        "train_rows": train_rows,
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    with REGISTRY_FILE.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return experiment_id


def save_model(experiment_id: str, pipeline: Any) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{experiment_id}.joblib"
    joblib.dump(pipeline, path)
    return path


def load_model(experiment_id: str) -> Any:
    path = MODELS_DIR / f"{experiment_id}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model artifact for {experiment_id}")
    return joblib.load(path)


def list_experiments() -> list[dict]:
    if not REGISTRY_FILE.exists():
        return []
    records = []
    with REGISTRY_FILE.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def champion_experiment(metric: str = "roc_auc") -> dict:
    """The experiment with the best CV score on `metric` (higher is better)."""
    records = list_experiments()
    if not records:
        raise RuntimeError("Registry is empty: run train.py first.")
    return max(records, key=lambda r: r["cv_scores"].get(metric, float("-inf")))
