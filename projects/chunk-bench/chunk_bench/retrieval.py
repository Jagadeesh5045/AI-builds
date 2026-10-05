"""TF-IDF retriever used as the fixed retrieval layer for the benchmark.

The retriever is deliberately held constant across chunking strategies: the
only thing that changes between runs is how documents were chunked, so any
difference in the metrics is attributable to chunking, not to retrieval.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunkers import Chunk


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Cannot build a retriever over zero chunks")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if not query or not query.strip():
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        ranked = sorted(
            zip(self.chunks, scores.tolist()), key=lambda pair: pair[1], reverse=True
        )
        return ranked[: max(1, k)]
