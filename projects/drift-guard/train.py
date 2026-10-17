"""Model training with cross-validated model selection.

Trains three candidate pipelines on the training split, each tuned with a
small GridSearchCV, and logs every run to the local experiment registry.
The champion (best CV ROC-AUC) is saved as a joblib artifact.

Usage:
    python train.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import data
import registry
from config import CHAMPION_METRIC, N_CV_FOLDS, RANDOM_STATE


def build_candidates() -> dict[str, tuple[Pipeline, dict]]:
    """(name -> (pipeline, param_grid)) for the model bake-off."""
    preprocess = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "logreg": (
            Pipeline(preprocess + [("clf", LogisticRegression(max_iter=2000,
                                                              random_state=RANDOM_STATE))]),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline(preprocess + [("clf", RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=-1))]),
            {"clf__n_estimators": [200, 400],
             "clf__max_depth": [None, 12],
             "clf__min_samples_leaf": [1, 4]},
        ),
        "hist_gb": (
            Pipeline(preprocess + [("clf", HistGradientBoostingClassifier(
                random_state=RANDOM_STATE))]),
            {"clf__learning_rate": [0.05, 0.15],
             "clf__max_depth": [None, 8],
             "clf__max_iter": [200, 400]},
        ),
    }


def main() -> str:
    df = data.load_dataset()
    X_train, X_test, y_train, y_test = data.split_data(df)
    features = data.feature_columns(df)

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}

    print(f"[train] {len(X_train)} train rows, {len(features)} features, "
          f"{N_CV_FOLDS}-fold stratified CV")
    for name, (pipeline, grid) in build_candidates().items():
        search = GridSearchCV(pipeline, grid, cv=cv, scoring=scoring,
                              refit=CHAMPION_METRIC, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        cv_scores = {m: float(search.cv_results_[f"mean_test_{m}"][search.best_index_])
                     for m in scoring}
        exp_id = registry.log_experiment(
            model_name=name,
            params=search.best_params_,
            cv_scores=cv_scores,
            train_rows=len(X_train),
            feature_names=features,
        )
        registry.save_model(exp_id, best)
        print(f"[train] {name:13s} CV roc_auc={cv_scores['roc_auc']:.4f} "
              f"f1={cv_scores['f1']:.4f} -> {exp_id}")

    champion = registry.champion_experiment(CHAMPION_METRIC)
    print(f"[train] champion: {champion['experiment_id']} "
          f"(CV {CHAMPION_METRIC}={champion['cv_scores'][CHAMPION_METRIC]:.4f})")
    return champion["experiment_id"]


if __name__ == "__main__":
    main()
