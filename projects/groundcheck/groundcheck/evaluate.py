"""Run groundcheck over the labelled cases and report detection quality."""
from .checker import check_answer
from .data import PASSAGES, CASES

def run_eval():
    rows, tp, fp, fn, tn = [], 0, 0, 0, 0
    for label, answer, idxs in CASES:
        passages = [PASSAGES[i] for i in idxs]
        rep = check_answer(answer, passages)
        flagged = rep.decision != "answer" or rep.unsupported > 0
        is_bad = label == "hallucinated"
        if flagged and is_bad: tp += 1
        elif flagged and not is_bad: fp += 1
        elif not flagged and is_bad: fn += 1
        else: tn += 1
        rows.append((label, rep.groundedness, rep.unsupported, rep.decision))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"rows": rows, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "cases": len(CASES)}
