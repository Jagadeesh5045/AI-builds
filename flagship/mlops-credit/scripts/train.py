"""Train and compare models, save artefacts, register both runs.

Champion = the model with the higher test ROC-AUC; the other is the
challenger. Run from the project root:  python scripts/train.py
"""

import json
import os
import sys

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlops_credit.data_gen import generate_credit_data  # noqa: E402
from mlops_credit.features import prepare  # noqa: E402
from mlops_credit.registry import append_entry, load_registry  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "credit.csv")
ARTIFACTS = os.path.join(ROOT, "artifacts")
REGISTRY = os.path.join(ROOT, "registry.json")


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
    }


def train(test_size: float = 0.2, seed: int = 42) -> dict:
    df = generate_credit_data(seed=seed)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    X = prepare(df)
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    candidates = {
        "LogisticRegression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000))]),
            {"max_iter": 1000, "solver": "lbfgs"},
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(n_estimators=200, random_state=seed),
            {"n_estimators": 200, "random_state": seed},
        ),
    }

    results = {}
    for name, (model, params) in candidates.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = {"model": model, "params": params, "metrics": metrics}
        joblib.dump(model, os.path.join(ARTIFACTS, f"{name}.joblib"))

    ranked = sorted(results.items(), key=lambda kv: kv[1]["metrics"]["roc_auc"],
                    reverse=True)
    champion_name, challenger_name = ranked[0][0], ranked[1][0]

    # Fresh registry for a clean demo run: keep only this run's entries.
    if os.path.exists(REGISTRY):
        os.remove(REGISTRY)
    for name, status in ((champion_name, "champion"),
                         (challenger_name, "challenger")):
        append_entry(REGISTRY, name, results[name]["params"],
                     results[name]["metrics"], status)

    summary = {
        "n_rows": len(df),
        "default_rate": round(float(y.mean()), 4),
        "test_size": len(y_test),
        "champion": champion_name,
        "challenger": challenger_name,
        "metrics": {n: r["metrics"] for n, r in results.items()},
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train()
