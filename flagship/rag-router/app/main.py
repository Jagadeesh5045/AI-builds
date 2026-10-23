"""FastAPI demo for rag-router.

POST /ask  -> route the query, run the chosen retrieval path, return the
              answer plus the chosen path, confidence and latency.
GET  /health, GET /routes for introspection.

Run:  uvicorn app.main:app --reload   (from the project root)
"""
import csv
import json
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_router.router import QueryRouter
from rag_router.retrieval import VectorRetriever, BM25Retriever, SQLRetriever
from rag_router.warehouse import load_warehouse
from rag_router import sqlgen

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MONTHS = {"january": 1, "february": 2, "march": 3,
          "april": 4, "may": 5, "june": 6}
REGIONS = ["north", "south", "east", "west"]

app = FastAPI(title="rag-router", version="1.0.0")
state = {}


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(5, ge=1, le=10)


@app.on_event("startup")
def startup():
    docs = [json.loads(l) for l in open(DATA / "corpus.jsonl") if l.strip()]
    with open(DATA / "products.csv") as f:
        products = list(csv.DictReader(f))
    product_names = [p["name"] for p in products]

    state["router"] = QueryRouter.load(ROOT / "artifacts" / "router.joblib")
    vec = VectorRetriever(); vec.fit(docs)
    bm25 = BM25Retriever(); bm25.fit(docs)
    state["retrievers"] = {"vector": vec, "keyword": bm25}
    con = load_warehouse(DATA / "products.csv", DATA / "orders.csv")
    state["sql"] = SQLRetriever(con, product_names, MONTHS, REGIONS)
    state["docs"] = len(docs)
    state["products"] = len(products)


@app.get("/health")
def health():
    return {"status": "ok", "docs": state.get("docs"),
            "products": state.get("products")}


@app.get("/routes")
def routes():
    return {"routes": [
        {"name": "vector",
         "description": "TF-IDF cosine search over the doc corpus; "
                        "best for open-ended conceptual questions"},
        {"name": "keyword",
         "description": "BM25 keyword search; best for exact product "
                        "names, SKUs and specs"},
        {"name": "sql",
         "description": "NL-to-SQL over the DuckDB orders warehouse; "
                        "best for aggregates like total revenue per month"},
    ]}


@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.perf_counter()
    route, confidence = state["router"].predict(req.query)

    answer, evidence, sql_used, fallback = None, None, None, None
    if route == "sql":
        try:
            res = state["sql"].answer(req.query)
            answer, sql_used = res["answer"], res["sql"]
            evidence = [{"type": "warehouse_row", "sql": res["sql"]}]
        except sqlgen.UnparseableQuery:
            # graceful fallback: conceptual search instead of failing
            proba = state["router"].predict_proba(req.query)
            route, confidence = "vector", proba["vector"]
            fallback = ("routed to 'sql' but the question was outside the "
                        "supported aggregate families; fell back to 'vector'")
    if route in ("vector", "keyword"):
        hits = state["retrievers"][route].search(req.query, k=req.k)
        answer = hits[0]["text"] if hits else "No matching documents found."
        evidence = [{"doc_id": h["doc_id"], "title": h["title"],
                     "score": round(h["score"], 4)} for h in hits]

    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "query": req.query,
        "path": route,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
        "answer": answer,
        "evidence": evidence,
        "sql": sql_used,
        "fallback": fallback,
    }
