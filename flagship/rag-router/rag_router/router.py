"""Query router: a scikit-learn classifier that routes each natural-language
query to the retrieval path best suited to it: 'vector', 'keyword' or 'sql'.

Features combine TF-IDF word n-grams with handcrafted lexical signals
(aggregate words, month names, regions, product/SKU mentions, digits).
"""
import re

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROUTES = ("vector", "keyword", "sql")

AGG_WORDS = {
    "total", "sum", "average", "avg", "mean", "count", "many", "highest",
    "most", "top", "best", "revenue", "orders", "units", "quantity", "sold",
    "typical",
}
QUESTION_WORDS = {"what", "which", "how", "when", "where", "who", "do", "does",
                  "is", "are", "can", "tell"}


def numeric_features(queries, product_names, skus, months, regions):
    """Handcrafted lexical signals, one row per query."""
    prod_names = [p.lower() for p in product_names]
    sku_set = {s.lower() for s in skus}
    feats = []
    for q in queries:
        ql = q.lower()
        words = set(re.findall(r"[a-z0-9]+", ql))
        feats.append([
            len(q),                                   # query length (chars)
            len(ql.split()),                          # query length (words)
            1.0 if re.search(r"\d", ql) else 0.0,     # contains a digit
            1.0 if words & AGG_WORDS else 0.0,        # aggregate intent words
            1.0 if any(m in ql for m in months) else 0.0,   # month mention
            1.0 if any(r in words for r in regions) else 0.0,  # region mention
            1.0 if any(p in ql for p in prod_names) else 0.0,  # product mention
            1.0 if words & sku_set else 0.0,          # SKU mention
            1.0 if ql.strip().endswith("?") else 0.0,
            1.0 if words & QUESTION_WORDS else 0.0,
        ])
    return np.asarray(feats, dtype=float)


class QueryRouter:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=1500, stop_words="english",
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, C=2.0, random_state=42)
        self.product_names, self.skus, self.months, self.regions = [], [], [], []

    def _numeric(self, queries):
        return self.scaler.transform(
            numeric_features(queries, self.product_names, self.skus,
                             self.months, self.regions))

    def _matrix(self, queries):
        return hstack([self.vectorizer.transform(queries),
                       csr_matrix(self._numeric(queries))]).tocsr()

    def fit(self, queries, labels, product_names, skus, months, regions):
        self.product_names, self.skus = list(product_names), list(skus)
        self.months, self.regions = list(months), list(regions)
        num = numeric_features(queries, self.product_names, self.skus,
                               self.months, self.regions)
        self.scaler.fit(num)
        X = hstack([self.vectorizer.fit_transform(queries),
                    csr_matrix(self.scaler.transform(num))]).tocsr()
        self.model.fit(X, labels)
        return self

    def predict(self, query):
        """Return (route, confidence)."""
        proba = self.predict_proba(query)
        best = max(proba, key=proba.get)
        return best, proba[best]

    def predict_proba(self, query):
        """Return {route: probability} for a single query."""
        X = self._matrix([query])
        proba = self.model.predict_proba(X)[0]
        return {c: float(p) for c, p in zip(self.model.classes_, proba)}

    def predict_many(self, queries):
        X = self._matrix(queries)
        proba = self.model.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return ([self.model.classes_[i] for i in idx],
                [float(proba[r, i]) for r, i in enumerate(idx)])

    def save(self, path):
        joblib.dump({
            "vectorizer": self.vectorizer, "scaler": self.scaler,
            "model": self.model, "product_names": self.product_names,
            "skus": self.skus, "months": self.months, "regions": self.regions,
        }, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        r = cls()
        r.vectorizer, r.scaler, r.model = d["vectorizer"], d["scaler"], d["model"]
        r.product_names, r.skus = d["product_names"], d["skus"]
        r.months, r.regions = d["months"], d["regions"]
        return r
