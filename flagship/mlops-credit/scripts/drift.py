"""Drift-triggered retraining demo.

Compares a drifted production batch against the training data with PSI
per feature. If any feature PSI exceeds the threshold, retrains the
champion model type on training + drifted data and registers a new
version. Run from the project root:  python scripts/drift.py
"""

import json
import os
import sys

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlops_credit.data_gen import (generate_credit_data,  # noqa: E402
                                   make_drifted_batch)
from mlops_credit.features import NUMERIC, prepare  # noqa: E402
from mlops_credit.psi import PSI_THRESHOLD, drift_detected, psi_report  # noqa: E402
from mlops_credit.registry import append_entry, champion, load_registry  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ARTIFACTS = os.path.join(ROOT, "artifacts")
REGISTRY = os.path.join(ROOT, "registry.json")


def build_model(model_type: str, seed: int = 42):
    if model_type == "LogisticRegression":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000))]), \
            {"max_iter": 1000, "solver": "lbfgs"}
    return RandomForestClassifier(n_estimators=200, random_state=seed), \
        {"n_estimators": 200, "random_state": seed}


def drift(seed: int = 42) -> dict:
    reference = generate_credit_data(seed=seed)
    drifted = make_drifted_batch(generate_credit_data(n_samples=2000, seed=99))
    report = {f: round(v, 4) for f, v in psi_report(reference, drifted, NUMERIC).items()}
    triggered = drift_detected(report, PSI_THRESHOLD)

    retrained = None
    if triggered:
        entries = load_registry(REGISTRY)
        champ = champion(entries)
        model_type = champ["model_type"] if champ else "RandomForestClassifier"
        model, params = build_model(model_type)
        combined = pd_concat(reference, drifted)
        model.fit(prepare(combined), combined["default"])
        joblib.dump(model, os.path.join(ARTIFACTS, f"{model_type}.joblib"))
        entry = append_entry(REGISTRY, model_type, params,
                             {"note": "retrained after drift trigger"}, "challenger")
        retrained = entry["version"]

    summary = {
        "psi_threshold": PSI_THRESHOLD,
        "psi": report,
        "drift_detected": triggered,
        "retrain_triggered": triggered,
        "retrained_version": retrained,
        "decision": ("RETRAINED and registered " + retrained) if triggered
                    else "No drift above threshold; no retraining",
    }
    print(json.dumps(summary, indent=2))
    return summary


def pd_concat(a, b):
    import pandas as pd
    return pd.concat([a, b], ignore_index=True)


if __name__ == "__main__":
    drift()
