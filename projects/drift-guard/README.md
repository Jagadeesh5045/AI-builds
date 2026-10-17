# Drift Guard

**ML training, evaluation, and deployment-readiness pipeline** — train competing models with cross-validated tuning, track every experiment in a local registry, evaluate the champion on a true holdout set, monitor incoming data for drift, and serve predictions over a REST API. One command runs everything reproducibly.

## The problem

Training a model is the easy part. The parts that break in production are the boring ones: nobody recorded which hyperparameters won, the "test accuracy" was measured on data the model already saw, nobody noticed the input distribution shifted, and the model file lives in someone's notebook. Drift Guard is a compact, dependency-light implementation of the full loop: **train → track → evaluate → monitor → serve**.

## What it does

- **Bake-off training** (`train.py`): 3 candidate pipelines (logistic regression, random forest, histogram gradient boosting), each median-imputation + scaling + classifier, tuned with 5-fold stratified `GridSearchCV`.
- **Experiment registry** (`registry.py`): every run appends a JSON record to `experiments.jsonl` (params, CV scores, timestamp); the fitted pipeline is saved as a `joblib` artifact. Experiment IDs are deterministic hashes of model + params, so re-runs are idempotent. Champion = best CV ROC-AUC.
- **Honest evaluation** (`evaluate.py`): the champion is scored once on the stratified holdout split it never saw. Writes `artifacts/reports/model_card.md` with accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrix, and a precision/recall-vs-threshold sweep.
- **Drift monitoring** (`drift.py`): per-feature Population Stability Index against the training baseline; flags batches with PSI > 0.25. The demo shows a clean batch passing (max PSI 0.23) and a shifted batch correctly flagged (`DRIFT`, the two moved features named).
- **Serving** (`serve.py`): FastAPI service with `/health`, `/predict` (validated feature dict → probability + label), and `/drift` (score an incoming batch against the training baseline).

## Real results (measured, not claimed)

Demo dataset: Wisconsin Breast Cancer, 569 rows × 30 features (sklearn-bundled public benchmark). 455 train / 114 holdout, stratified, seed 42. `python run_all.py` reproduces these exactly.

| Model | CV ROC-AUC | CV F1 |
|---|---|---|
| Logistic regression (C=1.0) — champion | **0.9959** | 0.9825 |
| Hist. gradient boosting | 0.9950 | 0.9755 |
| Random forest | 0.9905 | 0.9683 |

Champion on the unseen holdout (n=114): **accuracy 0.9825, F1 0.9861, ROC-AUC 0.9954**, confusion matrix 41/1/1/71. The honest lesson worth keeping: a tuned linear model beat both tree ensembles here. Complexity is not a strategy.

Drift check: clean batch → `OK` (max PSI 0.227); batch with two features shifted +1.5σ → `DRIFT`, flagging exactly `mean radius` and `mean texture` (max PSI 6.88).

## How to run

```bash
pip install -r requirements.txt

python run_all.py        # train all candidates, evaluate champion, write model card
python drift.py          # demo the drift check on clean vs shifted batches
python -m pytest tests/ -q   # 8 tests: validation, registry, drift, end-to-end train

uvicorn serve:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"mean radius": 14.2, ... }}'   # all 30 features
```

Swap the dataset by editing `data.load_dataset()` — the rest of the pipeline is dataset-agnostic.

## Tech stack

Python, scikit-learn (pipelines, GridSearchCV, metrics), pandas, NumPy, FastAPI + Pydantic + Uvicorn, joblib, pytest. No MLflow server, no cloud: the registry is JSONL + files, so the whole thing runs on a laptop.

## Project layout

```
drift-guard/
├── config.py      # seeds, paths, thresholds (reproducibility lives here)
├── data.py        # dataset loading, validation, stratified split
├── train.py       # 3-model bake-off + GridSearchCV
├── registry.py    # JSONL experiment log + joblib model store
├── evaluate.py    # holdout metrics + Markdown model card
├── drift.py       # PSI drift monitoring + shifted-batch demo
├── serve.py       # FastAPI inference + drift endpoints
├── run_all.py     # end-to-end entry point
└── tests/         # 8 unit tests
```

## Limits

Demo/benchmark pipeline, not a medical device: the dataset is a classic ML benchmark, not a clinical validation set. PSI thresholds are rules of thumb, tune them per domain.
