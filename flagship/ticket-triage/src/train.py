"""Train the ticket triage models.

Category classifier:  TF-IDF (word + bigrams) + LogisticRegression, calibrated.
Priority scorer:      TF-IDF + LogisticRegression, assisted by rule-based priority
                      signals blended in as extra features.

Saves: models/category_model.joblib, models/priority_model.joblib,
       models/vectorizer.joblib, models/metrics.json, models/baseline_distribution.json

Run: python src/train.py [--csv data/tickets.csv] [--test-size 0.2] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PRIORITIES = ["low", "medium", "high", "urgent"]

# Rule-based priority signals: patterns that genuinely indicate urgency.
# These are added as extra features to the priority model, not a replacement for it.
_RULE_PATTERNS = {
    "urgent": [r"\bunacceptable\b", r"\bescalat", r"\bformal complaint\b", r"\brude\b",
               r"\bhung up\b", r"\bwithout consent\b", r"\bcharged twice\b", r"\bnever received\b",
               r"\bsomeone logged into my account\b", r"\block.*out\b", r"\bfraud", r"\bscamm?ed\b",
               r"\bnot working\b", r"\bsite is down\b", r"\b503\b", r"\boutage\b"],
    "high": [r"\brefund\b", r"\bcrash", r"\bfaulty\b", r"\bbroken\b", r"\bdamaged\b",
             r"\bwrong address\b", r"\bduplicate charge\b", r"\bnot.*load\b", r"\bfailed\b",
             r"\bdoes not fit\b", r"\bwont power on\b", r"\bwill not power on\b", r"\bmissing\b"],
    "low": [r"\bjust wanted to say\b", r"\bthanks\b", r"\bgreat work\b", r"\blove the\b",
            r"\bsuggestion\b", r"\bsmall idea\b", r"\bfive stars\b", r"\bquestion\b",
            r"\bbefore i buy\b", r"\bhow do i\b.*\bchange\b"],
}


def rule_features(texts: list[str]) -> np.ndarray:
    """Binary features: does the text match urgent / high / low signal patterns?"""
    feats = np.zeros((len(texts), 3))
    for i, t in enumerate(texts):
        low_t = t.lower()
        for j, level in enumerate(["urgent", "high", "low"]):
            feats[i, j] = 1.0 if any(re.search(p, low_t) for p in _RULE_PATTERNS[level]) else 0.0
    return feats


def train(csv_path: Path, models_dir: Path, test_size: float = 0.2, seed: int = 42) -> dict:
    df = pd.read_csv(csv_path)
    assert {"text", "category", "priority"}.issubset(df.columns)

    X_train_t, X_test_t, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        df["text"].tolist(),
        df["category"].tolist(),
        df["priority"].tolist(),
        test_size=test_size,
        random_state=seed,
        stratify=df["category"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=8000,
    )
    Xtr = vectorizer.fit_transform(X_train_t)
    Xte = vectorizer.transform(X_test_t)

    # ---- category model: calibrated logistic regression ----
    cat_base = LogisticRegression(max_iter=1000, C=2.0, random_state=seed)
    cat_model = CalibratedClassifierCV(
        cat_base, method="sigmoid", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    )
    cat_model.fit(Xtr, y_cat_train)
    cat_pred = cat_model.predict(Xte)
    cat_proba = cat_model.predict_proba(Xte)

    # ---- priority model: TF-IDF + rule-signal features ----
    Rtr = csr_matrix(rule_features(X_train_t))
    Rte = csr_matrix(rule_features(X_test_t))
    pri_model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed,
                                   class_weight="balanced")
    pri_model.fit(hstack([Xtr, Rtr]), y_pri_train)
    pri_pred = pri_model.predict(hstack([Xte, Rte]))

    cat_acc = float(accuracy_score(y_cat_test, cat_pred))
    cat_f1 = float(f1_score(y_cat_test, cat_pred, average="macro"))
    pri_acc = float(accuracy_score(y_pri_test, pri_pred))
    pri_f1 = float(f1_score(y_pri_test, pri_pred, average="macro"))

    print("=== CATEGORY CLASSIFIER ===")
    print(f"accuracy={cat_acc:.4f}  macro-F1={cat_f1:.4f}")
    print(classification_report(y_cat_test, cat_pred, digits=3))
    print("=== PRIORITY SCORER ===")
    print(f"accuracy={pri_acc:.4f}  macro-F1={pri_f1:.4f}")
    print(classification_report(y_pri_test, pri_pred, digits=3))
    mean_conf = float(np.mean(np.max(cat_proba, axis=1)))
    print(f"mean predicted confidence (category): {mean_conf:.4f}")

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, models_dir / "vectorizer.joblib")
    joblib.dump(cat_model, models_dir / "category_model.joblib")
    joblib.dump(pri_model, models_dir / "priority_model.joblib")

    baseline = pd.Series(y_cat_train).value_counts(normalize=True).sort_index().to_dict()
    (models_dir / "baseline_distribution.json").write_text(
        json.dumps({k: float(v) for k, v in baseline.items()}, indent=2)
    )

    metrics = {
        "n_train": len(X_train_t),
        "n_test": len(X_test_t),
        "seed": seed,
        "category": {
            "accuracy": round(cat_acc, 4),
            "macro_f1": round(cat_f1, 4),
            "mean_confidence": round(mean_conf, 4),
            "report": classification_report(y_cat_test, cat_pred, digits=3, output_dict=True),
        },
        "priority": {
            "accuracy": round(pri_acc, 4),
            "macro_f1": round(pri_f1, 4),
            "report": classification_report(y_pri_test, pri_pred, digits=3, output_dict=True),
        },
    }
    # round report floats for readability
    for m in ("category", "priority"):
        for k, v in metrics[m]["report"].items():
            if isinstance(v, dict):
                metrics[m]["report"][k] = {kk: round(vv, 3) for kk, vv in v.items()}
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved models + metrics to {models_dir}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ticket triage models.")
    parser.add_argument("--csv", type=str, default="data/tickets.csv")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(Path(args.csv), Path(args.models_dir), args.test_size, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
