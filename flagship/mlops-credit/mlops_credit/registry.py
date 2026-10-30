"""Local JSON model registry.

Every training run appends one entry per model: version, model type,
hyperparameters, test metrics, timestamp and status. The promotion rule
lives here so train, canary and drift all share one decision policy.
"""

import json
import os
from datetime import datetime, timezone

PROMOTION_MARGIN = 0.02  # challenger must beat champion ROC-AUC by this much


def load_registry(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_registry(path: str, entries: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def next_version(entries: list) -> str:
    return f"v{len(entries) + 1}"


def append_entry(path: str, model_type: str, hyperparameters: dict,
                 metrics: dict, status: str) -> dict:
    entries = load_registry(path)
    entry = {
        "version": next_version(entries),
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    entries.append(entry)
    save_registry(path, entries)
    return entry


def champion(entries: list) -> dict | None:
    champs = [e for e in entries if e["status"] == "champion"]
    return champs[-1] if champs else None


def should_promote(champion_roc_auc: float, challenger_roc_auc: float,
                   margin: float = PROMOTION_MARGIN) -> bool:
    """Promote the challenger only if it beats the champion by `margin`."""
    return (challenger_roc_auc - champion_roc_auc) > margin


def promote(path: str, challenger_version: str) -> list:
    """Demote the current champion and promote the named challenger."""
    entries = load_registry(path)
    for e in entries:
        if e["status"] == "champion":
            e["status"] = "demoted"
        if e["version"] == challenger_version:
            e["status"] = "champion"
    save_registry(path, entries)
    return entries
