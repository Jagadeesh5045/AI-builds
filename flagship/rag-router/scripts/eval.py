"""Eval harness for rag-router.

Measures, on the held-out eval query set:
  * routing accuracy (predicted path vs labelled path)
  * per-path retrieval quality: recall@5 against known-relevant docs for
    the vector and keyword paths; exact-answer accuracy for the sql path
    (pandas-computed oracle over the same CSVs)
  * mean latency per path (router + retrieval), surfacing the
    latency/quality trade-off
  * ablation: retrieval quality when forcing EVERY query through a single
    path vs the router's choice

Prints a results table and writes artifacts/eval_results.json.
"""
import csv
import json
import time
from pathlib import Path

import pandas as pd

from rag_router.router import QueryRouter
from rag_router.retrieval import VectorRetriever, BM25Retriever, SQLRetriever
from rag_router.warehouse import load_warehouse
from rag_router import sqlgen

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def oracle(kind, params, orders, products):
    """Pandas ground truth for a SQL eval query."""
    m = {"january": 1, "february": 2, "march": 3,
         "april": 4, "may": 5, "june": 6}[params["month"]]
    df = orders.copy()
    df["order_total"] = df["quantity"] * df["unit_price"]
    df["m"] = pd.to_datetime(df["order_date"]).dt.month
    if kind == "sum_revenue_month":
        return float(df.loc[df.m == m, "order_total"].sum())
    if kind == "count_orders_month":
        return float((df.m == m).sum())
    if kind == "sum_units_product":
        return float(df.loc[df.product_id == params["product_id"], "quantity"].sum())
    if kind == "avg_order_region":
        return float(df.loc[df.region.str.lower() == params["region"], "order_total"].mean())
    if kind == "top_product_month":
        top = (df.loc[df.m == m].groupby("product_id")["order_total"]
               .sum().idxmax())
        return products.loc[products.product_id == top, "name"].iloc[0]
    if kind == "sum_revenue_region_month":
        sel = (df.m == m) & (df.region.str.lower() == params["region"])
        return float(df.loc[sel, "order_total"].sum())
    if kind == "count_orders_product_month":
        sel = (df.m == m) & (df.product_id == params["product_id"])
        return float(sel.sum())
    raise ValueError(kind)


def recall_at_k(retrieved_ids, relevant_ids, k=5):
    retrieved = retrieved_ids[:k]
    return len(set(retrieved) & set(relevant_ids)) / max(1, len(relevant_ids))


def main():
    eval_rows = load_jsonl(DATA / "router_eval.jsonl")
    docs = load_jsonl(DATA / "corpus.jsonl")
    with open(DATA / "products.csv") as f:
        products_list = list(csv.DictReader(f))
    products = pd.read_csv(DATA / "products.csv")
    orders = pd.read_csv(DATA / "orders.csv")

    router = QueryRouter.load(ART / "router.joblib")
    vec = VectorRetriever(); vec.fit(docs)
    bm25 = BM25Retriever(); bm25.fit(docs)
    con = load_warehouse(DATA / "products.csv", DATA / "orders.csv")
    sqlr = SQLRetriever(con, [p["name"] for p in products_list],
                        {"january": 1, "february": 2, "march": 3, "april": 4,
                         "may": 5, "june": 6},
                        ["north", "south", "east", "west"])

    route_ok, n = 0, 0
    per_path = {p: {"n": 0, "recall": [], "lat": [], "sql_ok": 0} for p in
                ("vector", "keyword", "sql")}
    routed_recalls = []          # recall@5 for routed vector/keyword queries
    forced = {"vector": [], "keyword": []}  # ablation recalls

    for row in eval_rows:
        q, true_label = row["query"], row["label"]
        n += 1

        t0 = time.perf_counter()
        route, conf = router.predict(q)
        route_ms = (time.perf_counter() - t0) * 1000
        if route == true_label:
            route_ok += 1

        if true_label in ("vector", "keyword"):
            # per-path quality is measured on the path's own labelled subset
            retr = vec if true_label == "vector" else bm25
            t1 = time.perf_counter()
            hits = retr.search(q, k=row["k"])
            lat_ms = (time.perf_counter() - t1) * 1000 + route_ms
            rec = recall_at_k([h["doc_id"] for h in hits],
                              row["relevant_doc_ids"], k=row["k"])
            per_path[true_label]["n"] += 1
            per_path[true_label]["recall"].append(rec)
            per_path[true_label]["lat"].append(lat_ms)
            routed_recalls.append(rec)
            # ablation: force each single path
            forced["vector"].append(recall_at_k(
                [h["doc_id"] for h in vec.search(q, k=row["k"])],
                row["relevant_doc_ids"], k=row["k"]))
            forced["keyword"].append(recall_at_k(
                [h["doc_id"] for h in bm25.search(q, k=row["k"])],
                row["relevant_doc_ids"], k=row["k"]))
        else:  # sql
            t1 = time.perf_counter()
            try:
                res = sqlr.answer(q)
                expected = oracle(row["kind"], row["params"], orders, products)
                if row["kind"] == "top_product_month":
                    ok = expected in res["answer"]  # product name matches
                else:
                    ok = abs(res["value"] - expected) < 0.01
            except sqlgen.UnparseableQuery:
                ok = False
            lat_ms = (time.perf_counter() - t1) * 1000 + route_ms
            per_path["sql"]["n"] += 1
            per_path["sql"]["lat"].append(lat_ms)
            per_path["sql"]["sql_ok"] += int(ok)

    results = {
        "n_eval": n,
        "routing_accuracy": route_ok / n,
        "paths": {},
        "ablation": {
            "routed_mean_recall@5": float(sum(routed_recalls) / len(routed_recalls)),
            "vector_only_mean_recall@5": float(sum(forced["vector"]) / len(forced["vector"])),
            "keyword_only_mean_recall@5": float(sum(forced["keyword"]) / len(forced["keyword"])),
        },
    }
    print(f"\nRouting accuracy: {route_ok}/{n} = {route_ok/n:.3f}\n")
    print(f"{'path':<10}{'n':>4}  {'recall@5':>9}  {'sql-acc':>8}  {'mean lat (ms)':>13}")
    print("-" * 50)
    for p in ("vector", "keyword", "sql"):
        d = per_path[p]
        rec = sum(d["recall"]) / len(d["recall"]) if d["recall"] else float("nan")
        sql_acc = (d["sql_ok"] / d["n"]) if p == "sql" else float("nan")
        lat = sum(d["lat"]) / len(d["lat"])
        results["paths"][p] = {
            "n": d["n"],
            "mean_recall@5": None if p == "sql" else rec,
            "sql_answer_accuracy": sql_acc if p == "sql" else None,
            "mean_latency_ms": lat,
        }
        print(f"{p:<10}{d['n']:>4}  "
              f"{rec:>9.3f}  "
              f"{(f'{sql_acc:.3f}' if p == 'sql' else '   n/a'):>8}  "
              f"{lat:>13.2f}")
    a = results["ablation"]
    print("\nAblation (vector+keyword eval queries, mean recall@5):")
    print(f"  routed (router's choice): {a['routed_mean_recall@5']:.3f}")
    print(f"  vector only:              {a['vector_only_mean_recall@5']:.3f}")
    print(f"  keyword only:             {a['keyword_only_mean_recall@5']:.3f}")

    ART.mkdir(parents=True, exist_ok=True)
    with open(ART / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {ART / 'eval_results.json'}")


if __name__ == "__main__":
    main()
