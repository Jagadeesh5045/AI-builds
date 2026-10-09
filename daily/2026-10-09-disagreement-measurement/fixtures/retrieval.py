"""Chunking and retrieval helpers for the demo codebase."""


def chunk_text(text, size=500, overlap=50):
    """Split text into overlapping chunks of roughly size characters."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def rank_chunks(chunks, scores):
    """Rank chunks by score in descending order, best first."""
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, score in ranked]


def reciprocal_rank_fusion(rank_lists, k=60):
    """Fuse several ranked lists with reciprocal rank fusion."""
    fused = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True)
