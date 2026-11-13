"""Fraud-risk model: Gradient Boosting over observable transaction signals.

Trained on the clean warehouse rows only. Evaluation is on a held-out
stratified test split, and every reported number is computed here, never
typed into the README by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["amount_gbp", "amount_log", "hour", "is_night", "is_foreign", "device_new", "velocity_10min", "day_of_week"]
CATEGORICAL = ["channel", "merchant_category"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.weekday
    out["is_night"] = ((out["hour"] >= 0) & (out["hour"] <= 5)).astype(int)
    out["amount_log"] = np.log1p(out["amount_gbp"])
    return out


def train_and_evaluate(clean: pd.DataFrame) -> dict:
    data = add_features(clean)
    X = data[NUMERIC + CATEGORICAL]
    y = data["is_fraud"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = Pipeline(
        [
            ("pre", pre),
            ("clf", HistGradientBoostingClassifier(random_state=42, max_iter=300, learning_rate=0.06)),
        ]
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Top-decile capture: what share of all test fraud sits in the riskiest 10%?
    order = np.argsort(-proba)
    k = max(1, int(0.10 * len(proba)))
    top_decile_capture = float(y_test.values[order[:k]].sum() / max(1, y_test.values.sum()))

    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc")
    importances = sorted(
        zip(NUMERIC + CATEGORICAL, perm.importances_mean), key=lambda t: -t[1]
    )

    scored_test = X_test.copy()
    scored_test["is_fraud"] = y_test.values
    scored_test["fraud_probability"] = proba
    scored_test["txn_id"] = data.loc[X_test.index, "txn_id"].values
    scored_test["timestamp"] = data.loc[X_test.index, "timestamp"].values
    scored_test["merchant_id"] = data.loc[X_test.index, "merchant_id"].values

    return {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_fraud": int(y_test.sum()),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "precision_at_0_5": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall_at_0_5": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_at_0_5": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "top_decile_fraud_capture": round(top_decile_capture, 4),
        "top_features": [(name, round(float(score), 4)) for name, score in importances[:5]],
        "scored_test": scored_test.sort_values("fraud_probability", ascending=False).reset_index(drop=True),
        "model": model,
    }
