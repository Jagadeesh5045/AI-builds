"""The three retrieval paths behind the router.

VectorRetriever  - TF-IDF cosine similarity over the doc corpus. An offline,
                   dependency-free stand-in for dense embedding search; swap
                   in sentence-transformers without changing the router.
BM25Retriever   - rank_bm25 keyword search; strongest on exact names/SKUs.
SQLRetriever    - rule-based NL->SQL (sqlgen) executed on the DuckDB
                   warehouse; for aggregate/factual questions over orders.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import sqlgen


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class VectorRetriever:
    name = "vector"

    def fit(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english",
                                         ngram_range=(1, 2),
                                         sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(
            [d["title"] + " " + d["text"] for d in docs])

    def search(self, query, k=5):
        qv = self.vectorizer.transform([query])
        scores = linear_kernel(qv, self.matrix).ravel()
        top = np.argsort(scores)[::-1][:k]
        return [{"doc_id": self.docs[i]["doc_id"],
                 "title": self.docs[i]["title"],
                 "text": self.docs[i]["text"],
                 "score": float(scores[i])} for i in top]


class BM25Retriever:
    name = "keyword"

    def fit(self, docs):
        self.docs = docs
        tokenized = [_tokens(d["title"] + " " + d["text"]) for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, k=5):
        scores = self.bm25.get_scores(_tokens(query))
        top = np.argsort(scores)[::-1][:k]
        return [{"doc_id": self.docs[i]["doc_id"],
                 "title": self.docs[i]["title"],
                 "text": self.docs[i]["text"],
                 "score": float(scores[i])} for i in top]


class SQLRetriever:
    name = "sql"

    def __init__(self, con, product_names, months, regions):
        self.con = con
        self.product_names = product_names
        self.months = months
        self.regions = regions

    def answer(self, query):
        """Run the NL->SQL pipeline. Returns a dict with the answer, or
        raises sqlgen.UnparseableQuery."""
        sql = sqlgen.to_sql(query, self.product_names, self.months, self.regions)
        df = self.con.execute(sql).fetchdf()
        if df.empty or df.iloc[0, 0] is None:
            raise sqlgen.UnparseableQuery("query returned no rows")
        if "product" in df.columns:  # top-product query
            product, revenue = df.iloc[0]["product"], float(df.iloc[0]["revenue"])
            answer = f"{product} (£{revenue:,.2f} revenue)"
            value = revenue
        else:
            value = float(df.iloc[0, 0])
            answer = f"£{value:,.2f}" if "revenue" in query.lower() or \
                "order value" in query.lower() else f"{value:,.0f}"
        return {"answer": answer, "value": value, "sql": sql}
