"""Drift check CLI: compare a new batch's predicted-category distribution
against the training baseline using Population Stability Index (PSI).

Usage:
    python src/drift.py --batch data/new_tickets.csv [--models-dir models]

The batch CSV must have a `text` column. Prints the PSI score and a verdict.
PSI < 0.1  -> no significant drift
PSI < 0.25 -> moderate drift, monitor
PSI >= 0.25 -> significant drift, investigate / retrain
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

EPS = 1e-6


def psi(expected: dict[str, float], actual: dict[str, float]) -> float:
    """Population Stability Index over shared category keys."""
    keys = sorted(set(expected) | set(actual))
    total = 0.0
    for k in keys:
        e = max(expected.get(k, 0.0), EPS)
        a = max(actual.get(k, 0.0), EPS)
        total += (a - e) * np.log(a / e)
    return float(total)


def verdict(score: float) -> str:
    if score < 0.1:
        return "NO SIGNIFICANT DRIFT — distributions are stable."
    if score < 0.25:
        return "MODERATE DRIFT — monitor; consider collecting fresh labels."
    return "SIGNIFICANT DRIFT — investigate and consider retraining."


def check(batch_csv: Path, models_dir: Path) -> tuple[float, str]:
    baseline = json.loads((models_dir / "baseline_distribution.json").read_text())
    vectorizer = joblib.load(models_dir / "vectorizer.joblib")
    cat_model = joblib.load(models_dir / "category_model.joblib")

    df = pd.read_csv(batch_csv)
    if "text" not in df.columns:
        raise ValueError("batch CSV must contain a 'text' column")
    texts = df["text"].astype(str).tolist()
    preds = cat_model.predict(vectorizer.transform(texts))
    actual = pd.Series(preds).value_counts(normalize=True).to_dict()
    actual = {k: float(v) for k, v in actual.items()}

    score = psi(baseline, actual)
    return score, verdict(score)


def main() -> int:
    parser = argparse.ArgumentParser(description="PSI drift check on a new ticket batch.")
    parser.add_argument("--batch", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    score, v = check(Path(args.batch), Path(args.models_dir))
    print(f"PSI score: {score:.4f}")
    print(f"Verdict: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
