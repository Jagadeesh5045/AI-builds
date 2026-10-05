"""chunk-bench: benchmark harness for RAG document-chunking strategies."""

from .chunkers import Chunk, fixed_size_chunks, heading_chunks, sentence_chunks
from .evaluate import STRATEGY_GRID, run_benchmark
from .metrics import contains_answer, context_precision_at_k, mean_reciprocal_rank, recall_at_k
from .retrieval import TfidfRetriever

__all__ = [
    "Chunk",
    "fixed_size_chunks",
    "heading_chunks",
    "sentence_chunks",
    "STRATEGY_GRID",
    "run_benchmark",
    "contains_answer",
    "context_precision_at_k",
    "mean_reciprocal_rank",
    "recall_at_k",
    "TfidfRetriever",
]

__version__ = "1.0.0"
