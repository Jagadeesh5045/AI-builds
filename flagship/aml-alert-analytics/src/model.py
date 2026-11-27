"""Suspicious alert model: gradient boosting with plain-English reasons.

The model scores each alert for the chance it is confirmed
suspicious. Alongside the headline metrics it explains individual
scores in words an analyst can act on, the amount, the customer risk
tier, the rule that fired, prior alerts and a new account, rather
than leaving a bare probability with no story attached.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = ["transaction_amount", "prior_alerts", "account_age_days"]
CATEGORICAL_FEATURES = ["channel", "alert_rule", "customer_risk_tier"]
TARGET = "confirmed_suspicious"


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES, dtype=float,
    )


def risk_reasons(row) -> list[str]:
    """Plain-English reasons one alert looks high risk."""
    reasons = []
    amount = float(row["transaction_amount"])
    if amount >= 50000:
        reasons.append(f"large transaction amount of {amount:,.2f}")
    elif amount >= 10000:
        reasons.append(f"transaction amount of {amount:,.2f}")
    if row["customer_risk_tier"] == "high":
        reasons.append("customer in the high risk tier")
    elif row["customer_risk_tier"] == "medium":
        reasons.append("customer in the medium risk tier")
    if row["alert_rule"] in ("sanctions_match", "structuring"):
        reasons.append(f"high-risk rule fired: {row['alert_rule']}")
    else:
        reasons.append(f"rule fired: {row['alert_rule']}")
    prior = int(row["prior_alerts"])
    if prior >= 2:
        reasons.append(f"{prior} prior alerts for this customer")
    elif prior == 1:
        reasons.append("1 prior alert for this customer")
    age = int(row["account_age_days"])
    if age < 180:
        reasons.append(f"new account, opened {age} days ago")
    if row["channel"] in ("wire", "correspondent"):
        reasons.append(f"{row['channel']} channel")
    return reasons


def train_model(frame: pd.DataFrame, seed: int = 42) -> dict:
    """Train, evaluate and explain the suspicious alert model."""
    features = _features(frame)
    train_x, test_x, train_y, test_y = train_test_split(
        features, frame[TARGET], test_size=0.25, random_state=seed,
        stratify=frame[TARGET],
    )
    test_rows = frame.loc[test_x.index]

    clf = GradientBoostingClassifier(
        n_estimators=180, learning_rate=0.06, max_depth=3, random_state=seed,
    )
    clf.fit(train_x, train_y)
    probabilities = clf.predict_proba(test_x)[:, 1]

    auc = roc_auc_score(test_y, probabilities)

    scored = pd.DataFrame({"probability": probabilities, "actual": test_y.values})
    decile_size = max(1, int(round(0.10 * len(scored))))
    top_decile = scored.nlargest(decile_size, "probability")
    total_positives = int(scored["actual"].sum())
    captured = int(top_decile["actual"].sum())
    capture_rate = round(captured / total_positives, 4) if total_positives else 0.0

    importances = sorted(
        zip(features.columns, clf.feature_importances_),
        key=lambda pair: pair[1], reverse=True,
    )

    ranked_test = test_rows.assign(probability=probabilities).sort_values(
        "probability", ascending=False
    )
    examples = []
    for _, row in ranked_test.head(5).iterrows():
        examples.append({
            "alert_id": row["alert_id"],
            "alert_rule": row["alert_rule"],
            "probability": round(float(row["probability"]), 3),
            "confirmed_suspicious": int(row[TARGET]),
            "reasons": risk_reasons(row),
        })

    return {
        "roc_auc": round(float(auc), 4),
        "test_alerts": int(len(scored)),
        "test_suspicious": total_positives,
        "top_decile_alerts": decile_size,
        "top_decile_capture_rate": capture_rate,
        "top_decile_captured": captured,
        "top_features": [
            {"feature": name, "importance": round(float(score), 4)}
            for name, score in importances[:6]
        ],
        "highest_risk_examples": examples,
    }
