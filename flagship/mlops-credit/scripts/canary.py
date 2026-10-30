"""Canary deploy demo: 90% champion traffic, 10% challenger traffic.

Scores a fresh simulated batch, compares ROC-AUC on each slice, and
applies the registry promotion rule. Run from the project root:
  python scripts/canary.py
"""

import json
import os
import sys

import joblib
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlops_credit.data_gen import generate_credit_data  # noqa: E402
from mlops_credit.features import prepare  # noqa: E402
from mlops_credit.registry import (champion, load_registry, promote,  # noqa: E402
                                   should_promote)

ROOT = os.path.join(os.path.dirname(__file__), "..")
ARTIFACTS = os.path.join(ROOT, "artifacts")
REGISTRY = os.path.join(ROOT, "registry.json")


def canary(seed: int = 123, margin: float = 0.02) -> dict:
    entries = load_registry(REGISTRY)
    champ = champion(entries)
    challengers = [e for e in entries if e["status"] == "challenger"]
    if champ is None or not challengers:
        raise RuntimeError("Registry needs a champion and a challenger; run train first.")
    challenger = challengers[-1]

    batch = generate_credit_data(n_samples=2000, seed=seed)
    X, y = prepare(batch), batch["default"]

    # Deterministic traffic split: every 10th request goes to the challenger.
    is_challenger = (batch.index % 10 == 9)
    champ_model = joblib.load(os.path.join(ARTIFACTS, f"{champ['model_type']}.joblib"))
    chall_model = joblib.load(os.path.join(ARTIFACTS, f"{challenger['model_type']}.joblib"))

    champ_auc = round(roc_auc_score(y[~is_challenger],
                                    champ_model.predict_proba(X[~is_challenger])[:, 1]), 4)
    chall_auc = round(roc_auc_score(y[is_challenger],
                                    chall_model.predict_proba(X[is_challenger])[:, 1]), 4)
    decision = should_promote(champ_auc, chall_auc, margin)
    if decision:
        promote(REGISTRY, challenger["version"])

    summary = {
        "batch_size": len(batch),
        "champion": champ["model_type"],
        "challenger": challenger["model_type"],
        "champion_traffic_share": 0.9,
        "challenger_traffic_share": 0.1,
        "champion_roc_auc": champ_auc,
        "challenger_roc_auc": chall_auc,
        "promotion_margin": margin,
        "decision": "PROMOTE challenger" if decision else "KEEP champion",
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    canary()
