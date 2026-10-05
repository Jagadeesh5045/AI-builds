"""Benchmark driver: chunks the corpus with every strategy in the grid,
indexes each chunking with the same TF-IDF retriever, scores every QA pair,
and aggregates the metrics.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .chunkers import Chunk, fixed_size_chunks, heading_chunks, sentence_chunks
from .corpus import Document
from .metrics import (
    contains_answer,
    context_precision_at_k,
    mean_reciprocal_rank,
    recall_at_k,
)
from .qa import QAPair
from .retrieval import TfidfRetriever

ChunkerFn = Callable[..., list[Chunk]]

# (label, chunker function, kwargs). Retrieval is identical for every row,
# so metric differences are attributable to chunking alone.
STRATEGY_GRID: list[tuple[str, ChunkerFn, dict]] = [
    ("fixed-500", fixed_size_chunks, {"size": 500, "overlap": 50, "strategy": "fixed-500"}),
    ("fixed-1000", fixed_size_chunks, {"size": 1000, "overlap": 100, "strategy": "fixed-1000"}),
    ("fixed-2000", fixed_size_chunks, {"size": 2000, "overlap": 200, "strategy": "fixed-2000"}),
    ("sentence-1000", sentence_chunks, {"max_chars": 1000, "overlap_sentences": 1,
                                       "strategy": "sentence-1000"}),
    ("sentence-2000", sentence_chunks, {"max_chars": 2000, "overlap_sentences": 1,
                                       "strategy": "sentence-2000"}),
    ("heading-1000", heading_chunks, {"max_chars": 1000, "strategy": "heading-1000"}),
    ("heading-2000", heading_chunks, {"max_chars": 2000, "strategy": "heading-2000"}),
]


def chunk_corpus(docs: list[Document], label: str, fn: ChunkerFn, kwargs: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(fn(doc.doc_id, doc.text, **kwargs))
    # Stamp the grid label on every chunk for traceability.
    for chunk in chunks:
        chunk.strategy = label
    return chunks


def evaluate_strategy(
    chunks: list[Chunk], pairs: list[QAPair], k: int = 5
) -> dict[str, float]:
    retriever = TfidfRetriever(chunks)
    recalls, mrrs, ctx_precs = [], [], []
    for pair in pairs:
        ranked = retriever.search(pair.question, k=max(k, 20))
        recalls.append(recall_at_k(ranked, pair.answer_span, k=k))
        mrrs.append(mean_reciprocal_rank(ranked, pair.answer_span))
        ctx_precs.append(context_precision_at_k(ranked, pair.doc_id, k=k))
    n = len(pairs)
    return {
        "recall@k": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "context_precision@k": sum(ctx_precs) / n,
        "n_chunks": float(len(chunks)),
        "avg_chunk_chars": float(sum(len(c.text) for c in chunks) / len(chunks)),
    }


def run_benchmark(
    docs: list[Document],
    pairs: list[QAPair],
    k: int = 5,
    grid: list[tuple[str, ChunkerFn, dict]] | None = None,
) -> pd.DataFrame:
    rows = []
    for label, fn, kwargs in grid or STRATEGY_GRID:
        chunks = chunk_corpus(docs, label, fn, kwargs)
        metrics = evaluate_strategy(chunks, pairs, k=k)
        rows.append({"strategy": label, **metrics})
    df = pd.DataFrame(rows).set_index("strategy")
    return df.sort_values("recall@k", ascending=False)


def format_table(df: pd.DataFrame) -> str:
    """Render the results as a plain-text table (no tabulate dependency)."""
    show = df.copy()
    for col in ("recall@k", "mrr", "context_precision@k"):
        show[col] = show[col].map(lambda v: f"{v:.3f}")
    show["n_chunks"] = show["n_chunks"].map(lambda v: f"{v:.0f}")
    show["avg_chunk_chars"] = show["avg_chunk_chars"].map(lambda v: f"{v:.0f}")
    widths = {c: max(len(c), show[c].astype(str).map(len).max()) for c in show.columns}
    header = "strategy".ljust(14) + "  " + "  ".join(c.ljust(widths[c]) for c in show.columns)
    lines = [header, "-" * len(header)]
    for strategy, row in show.iterrows():
        lines.append(
            str(strategy).ljust(14) + "  " + "  ".join(str(row[c]).ljust(widths[c]) for c in show.columns)
        )
    return "\n".join(lines)
