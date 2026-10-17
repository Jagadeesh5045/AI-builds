"""FastAPI inference service for the champion model.

Endpoints:
    GET  /health    service + model status
    POST /predict   {"features": {name: value, ...}} -> label + probability
    POST /drift     {"rows": [{name: value, ...}]} -> PSI drift verdict vs training baseline

The training baseline (for drift) is recomputed from the training split at
startup, so the service is self-contained: it only needs the registry.

Usage:
    uvicorn serve:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import data
import drift as drift_mod
import registry
from config import API_HOST, API_PORT, CHAMPION_METRIC


class PredictRequest(BaseModel):
    features: dict[str, float]


class DriftRequest(BaseModel):
    rows: list[dict[str, float]]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    exp = registry.champion_experiment(CHAMPION_METRIC)
    df = data.load_dataset()
    X_train, _, _, _ = data.split_data(df)
    state["experiment_id"] = exp["experiment_id"]
    state["model_name"] = exp["model"]
    state["pipeline"] = registry.load_model(exp["experiment_id"])
    state["features"] = data.feature_columns(df)
    state["X_train"] = X_train
    yield


app = FastAPI(title="drift-guard inference", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok",
            "experiment_id": state.get("experiment_id"),
            "model": state.get("model_name")}


def _frame(payload: dict[str, float]) -> pd.DataFrame:
    missing = [c for c in state["features"] if c not in payload]
    if missing:
        raise HTTPException(status_code=422,
                            detail=f"missing features: {missing[:5]}")
    return pd.DataFrame([{c: float(payload[c]) for c in state["features"]}])


@app.post("/predict")
def predict(req: PredictRequest):
    frame = _frame(req.features)
    proba = float(state["pipeline"].predict_proba(frame)[0, 1])
    return {"experiment_id": state["experiment_id"],
            "probability": round(proba, 4),
            "label": int(proba >= 0.5)}


@app.post("/drift")
def drift(req: DriftRequest):
    if not req.rows:
        raise HTTPException(status_code=422, detail="no rows supplied")
    batch = pd.DataFrame([{c: float(r.get(c, float("nan")))
                           for c in state["features"]} for r in req.rows])
    if batch.isna().any().any():
        raise HTTPException(status_code=422, detail="rows contain missing features")
    return drift_mod.check_batch(state["X_train"], batch)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
