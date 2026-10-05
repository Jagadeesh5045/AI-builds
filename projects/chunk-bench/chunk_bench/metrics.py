"""Evaluation metrics.

Relevance is defined by *answer containment*: a chunk is relevant to a
question if it contains the question's gold answer span (normalised). This is
the standard cheap proxy used when benchmarking chunking strategies, because
it directly measures whether the retriever can surface the exact passage the
answer lives in.
"""
from __future__ import annotations

import re
import string

from .chunkers import Chunk


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def contains_answer(chunk_text: str, answer_span: str) -> bool:
    return normalize(answer_span) in normalize(chunk_text)


def _relevant_flags(ranked: list[tuple[Chunk, float]], answer_span: str) -> list[bool]:
    return [contains_answer(chunk.text, answer_span) for chunk, _ in ranked]


def recall_at_k(ranked: list[tuple[Chunk, float]], answer_span: str, k: int = 5) -> float:
    """1.0 if any of the top-k chunks contains the answer, else 0.0."""
    return 1.0 if any(_relevant_flags(ranked[:k], answer_span)) else 0.0


def mean_reciprocal_rank(ranked: list[tuple[Chunk, float]], answer_span: str) -> float:
    """1/rank of the first chunk containing the answer (0.0 if never found)."""
    for rank, relevant in enumerate(_relevant_flags(ranked, answer_span), start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def context_precision_at_k(
    ranked: list[tuple[Chunk, float]], doc_id: str, k: int = 5
) -> float:
    """Fraction of the top-k chunks that came from the correct document."""
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for chunk, _ in top if chunk.doc_id == doc_id) / len(top)
