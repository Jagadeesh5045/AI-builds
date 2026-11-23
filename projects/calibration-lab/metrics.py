"""Evaluation metrics for probabilistic classification."""

import numpy as np


def roc_auc(y, p):
    """Rank-based ROC-AUC with tie handling. Returns nan if one class only."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allp = np.concatenate([pos, neg])
    for v in np.unique(allp):
        mask = allp == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    rank_pos = ranks[: len(pos)].sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def log_loss(y, p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            ece += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def confusion(y, p, threshold):
    pred = (np.asarray(p) >= threshold).astype(int)
    y = np.asarray(y)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
    }


def expected_cost(y, p, threshold, cost_fn=450.0, cost_fp=60.0):
    c = confusion(y, p, threshold)
    return float(c["fn"] * cost_fn + c["fp"] * cost_fp)
