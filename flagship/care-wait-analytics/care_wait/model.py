"""Breach-risk model: predicts whether a referral will wait longer than
18 weeks (126 days) at the point of referral, from features a service
would know on day one. Explainability is grouped feature importance,
so a manager sees 'specialty' rather than a list of dummy columns."""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURE_GROUPS = {
    "specialty": "specialty",
    "priority": "priority",
    "age_band": "age_band",
}


def build_features(df):
    d = df.copy()
    d["referral_month"] = pd.to_datetime(d["referral_date"]).dt.month
    X = pd.get_dummies(
        d[["specialty", "priority", "age_band", "imd_quintile", "referral_month"]],
        columns=["specialty", "priority", "age_band"],
        dtype=float,
    )
    return X, d["breach_18wk"].astype(int)


def train_breach_model(df, seed=42):
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    importance = {}
    for col, imp in zip(X.columns, model.feature_importances_):
        if col.startswith("imd"):
            group = "imd_quintile"
        elif col.startswith("referral"):
            group = "referral_month"
        elif col.startswith("age_band"):
            group = "age_band"
        elif col.startswith("specialty"):
            group = "specialty"
        elif col.startswith("priority"):
            group = "priority"
        else:
            group = col
        importance[group] = round(float(importance.get(group, 0.0) + imp), 4)
    importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
    return {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_breach_rate_pct": round(100.0 * float(y_test.mean()), 1),
        "feature_importance_grouped": importance,
        "top_feature": next(iter(importance)),
    }
