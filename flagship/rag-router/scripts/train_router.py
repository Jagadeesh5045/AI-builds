"""Train the query router classifier and save it to artifacts/router.joblib.

Reports held-out validation accuracy on a stratified 80/20 split of the
training queries.
"""
import csv
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from rag_router.router import QueryRouter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ART.mkdir(parents=True, exist_ok=True)
    train = load_jsonl(DATA / "router_train.jsonl")
    queries = [r["query"] for r in train]
    labels = [r["label"] for r in train]

    with open(DATA / "products.csv") as f:
        products = list(csv.DictReader(f))
    product_names = [p["name"] for p in products]
    skus = [p["sku"] for p in products]
    months = {"january": 1, "february": 2, "march": 3, "april": 4,
              "may": 5, "june": 6}
    regions = ["north", "south", "east", "west"]

    Xtr, Xva, ytr, yva = train_test_split(
        queries, labels, test_size=0.2, random_state=42, stratify=labels)
    router = QueryRouter().fit(Xtr, ytr, product_names, skus, months, regions)
    pred, _ = router.predict_many(Xva)
    acc = accuracy_score(yva, pred)
    print(f"train size={len(Xtr)}  validation size={len(Xva)}")
    print(f"validation accuracy: {acc:.3f}")
    print(classification_report(yva, pred, digits=3))

    full = QueryRouter().fit(queries, labels, product_names, skus, months, regions)
    out = ART / "router.joblib"
    full.save(out)
    print(f"saved router -> {out}")


if __name__ == "__main__":
    main()
