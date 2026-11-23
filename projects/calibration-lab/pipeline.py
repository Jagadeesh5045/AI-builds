"""End-to-end pipeline: train, calibrate, choose a threshold, audit slices.

Run:  python pipeline.py
Writes artifacts/metrics.json, artifacts/model_card.json and report.md.
Costs are stated assumptions for the threshold stage, not measured prices:
a missed readmission (false negative) is costed at 450 pounds of avoidable
care, an unnecessary follow-up (false positive) at 60 pounds.
"""

import json
import os

import numpy as np

from calibration import IsotonicCalibrator, PlattScaler
from data import FEATURES, generate_readmissions, stratified_split
from metrics import (brier, confusion, expected_calibration_error,
                     expected_cost, log_loss, roc_auc)
from model import LogisticModel

COST_FN = 450.0
COST_FP = 60.0
HERE = os.path.dirname(os.path.abspath(__file__))


def score_block(y, p):
    return {
        "roc_auc": round(roc_auc(y, p), 4),
        "brier": round(brier(y, p), 4),
        "log_loss": round(log_loss(y, p), 4),
        "ece": round(expected_calibration_error(y, p), 4),
    }


def choose_threshold(y, p):
    best_t, best_c = 0.5, float("inf")
    for t in np.arange(0.05, 0.951, 0.01):
        c = expected_cost(y, p, float(t), COST_FN, COST_FP)
        if c < best_c:
            best_c, best_t = c, float(t)
    return round(best_t, 2), best_c


def slice_report(X, y, p, features):
    slices = {}
    age = X[:, features.index("age")]
    bands = {"age<65": age < 65, "age65-79": (age >= 65) & (age < 80), "age80+": age >= 80}
    prior = X[:, features.index("prior_admissions")]
    bands["prior0"] = prior == 0
    bands["prior1"] = prior == 1
    bands["prior2+"] = prior >= 2
    for name, m in bands.items():
        if m.sum() < 30:
            continue
        slices[name] = {
            "n": int(m.sum()),
            "prevalence": round(float(y[m].mean()), 4),
            "roc_auc": round(roc_auc(y[m], p[m]), 4),
            "brier": round(brier(y[m], p[m]), 4),
        }
    return slices


def run(out_dir=None, n=3000, seed=42):
    out_dir = out_dir or os.path.join(HERE, "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    X, y, features = generate_readmissions(n=n, seed=seed)
    tr, va, te = stratified_split(y)
    model = LogisticModel().fit(X[tr], y[tr])

    raw_va = model.predict_proba(X[va])
    raw_te = model.predict_proba(X[te])

    platt = PlattScaler().fit(raw_va, y[va])
    iso = IsotonicCalibrator().fit(raw_va, y[va])
    candidates = {
        "raw": raw_va,
        "platt": platt.predict(raw_va),
        "isotonic": iso.predict(raw_va),
    }
    val_scores = {k: score_block(y[va], v) for k, v in candidates.items()}
    # Selection rule: lowest validation log loss. Log loss is a proper
    # scoring rule that punishes extreme wrong probabilities, which is
    # exactly the failure mode of isotonic steps fitted on 600 rows:
    # isotonic won validation Brier and ECE here, but its 0 and 1 steps
    # cost far more in log loss than Platt scaling. See report.md.
    chosen = min(candidates, key=lambda k: val_scores[k]["log_loss"])
    calibrators = {"raw": None, "platt": platt, "isotonic": iso}
    cal = calibrators[chosen]
    test_p = raw_te if cal is None else cal.predict(raw_te)
    val_p = candidates[chosen]

    threshold, val_cost = choose_threshold(y[va], val_p)
    test_conf = confusion(y[te], test_p, threshold)
    test_cost = expected_cost(y[te], test_p, threshold, COST_FN, COST_FP)
    default_cost = expected_cost(y[te], test_p, 0.5, COST_FN, COST_FP)

    results = {
        "dataset": {"n": int(n), "prevalence": round(float(y.mean()), 4),
                    "train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "validation": val_scores,
        "chosen_calibration": chosen,
        "test_raw": score_block(y[te], raw_te),
        "test_calibrated": score_block(y[te], test_p),
        "threshold": {"chosen": threshold, "cost_fn": COST_FN, "cost_fp": COST_FP,
                      "val_cost": round(val_cost, 2),
                      "test_cost_at_chosen": round(test_cost, 2),
                      "test_cost_at_0.5": round(default_cost, 2),
                      "test_confusion": test_conf},
        "slices_test": slice_report(X[te], y[te], test_p, features),
    }

    card = {
        "model": "logistic-regression-numpy-gd",
        "version": "1.0.0",
        "features": features,
        "mean": model.mean_.tolist(),
        "std": model.std_.tolist(),
        "weights": model.w_.tolist(),
        "bias": model.b_,
        "calibration": chosen,
        "platt": None if platt is None else {"a": platt.a_, "b": platt.b_},
        "isotonic": None if iso.x_ is None else
                    {"x": iso.x_.tolist(), "y": iso.y_.tolist()},
        "threshold": threshold,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "model_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    write_report(results, os.path.join(HERE, "report.md"))
    return results


def write_report(r, path):
    t = r["threshold"]
    worst = max(r["slices_test"].items(), key=lambda kv: kv[1]["brier"])
    lines = [
        "# Calibration Lab: results",
        "",
        f"Dataset: {r['dataset']['n']} synthetic discharges, prevalence "
        f"{r['dataset']['prevalence']:.3f}. Split {r['dataset']['train']} train, "
        f"{r['dataset']['val']} validation, {r['dataset']['test']} test.",
        "",
        "## Test set, raw vs calibrated",
        "",
        f"Raw: ROC-AUC {r['test_raw']['roc_auc']}, Brier {r['test_raw']['brier']}, "
        f"log loss {r['test_raw']['log_loss']}, ECE {r['test_raw']['ece']}.",
        f"Calibrated ({r['chosen_calibration']}): ROC-AUC "
        f"{r['test_calibrated']['roc_auc']}, Brier {r['test_calibrated']['brier']}, "
        f"log loss {r['test_calibrated']['log_loss']}, ECE {r['test_calibrated']['ece']}.",
        "",
        "## Threshold",
        "",
        f"Chosen on validation at a false-negative cost of £{t['cost_fn']:.0f} and a "
        f"false-positive cost of £{t['cost_fp']:.0f}: {t['chosen']}. "
        f"Test cost at the chosen threshold: £{t['test_cost_at_chosen']:,.2f} for the "
        f"test split, versus £{t['test_cost_at_0.5']:,.2f} at the default 0.5. "
        f"Test confusion at the chosen threshold: {t['test_confusion']}.",
        "",
        "## Slices (test, calibrated)",
        "",
    ]
    for name, s in r["slices_test"].items():
        lines.append(f"- {name}: n={s['n']}, prevalence {s['prevalence']}, "
                     f"ROC-AUC {s['roc_auc']}, Brier {s['brier']}.")
    lines += ["", f"Worst slice by Brier: {worst[0]} at {worst[1]['brier']}.", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2))
