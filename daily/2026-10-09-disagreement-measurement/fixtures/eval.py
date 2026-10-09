"""Retrieval evaluation metrics for the demo codebase."""


def recall_at_k(retrieved, relevant, k):
    """Fraction of relevant docs found in the top-k retrieved."""
    top_k = set(retrieved[:k])
    hit = len(top_k.intersection(relevant))
    return hit / len(relevant) if relevant else 0.0


def precision_at_k(retrieved, relevant, k):
    """Fraction of the top-k retrieved docs that are relevant."""
    top_k = retrieved[:k]
    hit = sum(1 for doc in top_k if doc in relevant)
    return hit / len(top_k) if top_k else 0.0
