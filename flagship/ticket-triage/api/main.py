"""FastAPI service for AI ticket triage.

Endpoints:
    POST /triage        {text} -> {category, confidence, priority, suggested_reply}
    POST /triage/batch  {texts: [...]} -> {results: [...]}
    GET  /health
    GET  /metrics       served metrics.json + drift verdict on recent request mix

Run: uvicorn api.main:app --reload
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.drift import psi
from src.templates import get_template
from src.train import rule_features

MODELS_DIR = ROOT / "models"
MAX_TEXT_LEN = 5000

_vectorizer = None
_cat_model = None
_pri_model = None
_baseline: dict[str, float] = {}
_request_categories: Counter = Counter()


def _load():
    global _vectorizer, _cat_model, _pri_model, _baseline
    _vectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
    _cat_model = joblib.load(MODELS_DIR / "category_model.joblib")
    _pri_model = joblib.load(MODELS_DIR / "priority_model.joblib")
    _baseline = json.loads((MODELS_DIR / "baseline_distribution.json").read_text())


def _features(texts: list[str]):
    X = _vectorizer.transform(texts)
    R = csr_matrix(rule_features(texts))
    return X, hstack([X, R])


def triage_text(text: str) -> dict:
    X, XR = _features([text])
    probs = _cat_model.predict_proba(X)[0]
    idx = int(probs.argmax())
    category = str(_cat_model.classes_[idx])
    confidence = round(float(probs[idx]), 4)
    priority = str(_pri_model.predict(XR)[0])
    _request_categories[category] += 1
    return {
        "category": category,
        "confidence": confidence,
        "priority": priority,
        "suggested_reply": get_template(category),
    }


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(title="ticket-triage", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _vectorizer is not None}


@app.post("/triage")
def triage(req: TriageRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    return triage_text(text)


@app.post("/triage/batch")
def triage_batch(req: BatchRequest):
    results = []
    for t in req.texts:
        t = (t or "").strip()
        if not t:
            results.append({"error": "empty text skipped"})
        elif len(t) > MAX_TEXT_LEN:
            results.append({"error": f"text exceeds {MAX_TEXT_LEN} chars, skipped"})
        else:
            results.append(triage_text(t))
    return {"results": results, "count": len(results)}


@app.get("/metrics")
def metrics():
    served = json.loads((MODELS_DIR / "metrics.json").read_text())
    drift = {"psi": None, "verdict": "no requests served yet"}
    if _request_categories:
        total = sum(_request_categories.values())
        actual = {k: v / total for k, v in _request_categories.items()}
        score = psi(_baseline, actual)
        drift = {
            "psi": round(score, 4),
            "verdict": (
                "no significant drift" if score < 0.1
                else "moderate drift" if score < 0.25
                else "significant drift"
            ),
            "requests_observed": total,
        }
    return {"served_metrics": served, "drift": drift}
