"""Readmission risk model: gradient boosting with plain-English reasons.

The model scores each discharged patient for 30-day readmission risk.
Alongside the headline metrics it explains individual scores in words a
discharge team can act on - prior admissions, comorbidities, emergency
admission, a long stay, a high-risk condition - rather than leaving a
bare probability with no story attached.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = ["length_of_stay_days", "prior_admissions",
                    "comorbidity_count", "deprivation_quintile"]
CATEGORICAL_FEATURES = ["age_band", "sex", "primary_condition",
                        "admission_type", "discharge_destination"]
TARGET = "readmitted_30d"


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES, dtype=float,
    )


def risk_reasons(row) -> list[str]:
    """Plain-English reasons one patient's risk is elevated."""
    reasons = []
    prior = int(row["prior_admissions"])
    if prior >= 2:
        reasons.append(f"{prior} prior admissions in the last year")
    elif prior == 1:
        reasons.append("1 prior admission in the last year")
    comorbidities = int(row["comorbidity_count"])
    if comorbidities >= 4:
        reasons.append(f"{comorbidities} comorbidities")
    if row["admission_type"] == "emergency":
        reasons.append("emergency admission")
    los = int(row["length_of_stay_days"])
    if los >= 10:
        reasons.append(f"long stay of {los} days")
    if row["primary_condition"] in ("heart failure", "COPD"):
        reasons.append(f"primary condition: {row['primary_condition']}")
    if row["age_band"] in ("75-84", "85+"):
        reasons.append(f"age band {row['age_band']}")
    if row["discharge_destination"] in ("care home", "home with care"):
        reasons.append(f"discharged to {row['discharge_destination']}")
    return reasons


def train_model(frame: pd.DataFrame, seed: int = 42) -> dict:
    """Train, evaluate and explain the readmission risk model."""
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

    # Top-decile capture: of all readmissions in the test set, what share
    # sit inside the highest-risk tenth of patients?
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
            "discharge_id": row["discharge_id"],
            "primary_condition": row["primary_condition"],
            "probability": round(float(row["probability"]), 3),
            "readmitted_30d": int(row[TARGET]),
            "reasons": risk_reasons(row),
        })

    return {
        "roc_auc": round(float(auc), 4),
        "test_patients": int(len(scored)),
        "test_readmissions": total_positives,
        "top_decile_patients": decile_size,
        "top_decile_capture_rate": capture_rate,
        "top_decile_captured": captured,
        "top_features": [
            {"feature": name, "importance": round(float(score), 4)}
            for name, score in importances[:6]
        ],
        "highest_risk_examples": examples,
    }
