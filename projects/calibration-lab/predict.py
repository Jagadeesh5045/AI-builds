"""Score one patient from the saved model card.

Usage: python predict.py '{"age": 78, "prior_admissions": 2,
"length_of_stay_days": 6, "comorbidity_count": 3, "medication_count": 9,
"follow_up_booked": 0, "weekend_discharge": 1}'
Missing features default to the training means stored in the card.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_card(path=None):
    path = path or os.path.join(HERE, "artifacts", "model_card.json")
    with open(path) as f:
        return json.load(f)


def apply_calibration(card, raw):
    method = card["calibration"]
    if method == "platt":
        a, b = card["platt"]["a"], card["platt"]["b"]
        p = min(max(raw, 1e-9), 1 - 1e-9)
        z = np.log(p / (1 - p))
        return float(1.0 / (1.0 + np.exp(-(a * z + b))))
    if method == "isotonic":
        xs = np.array(card["isotonic"]["x"])
        ys = np.array(card["isotonic"]["y"])
        idx = int(np.clip(np.searchsorted(xs, raw, side="left"), 0, len(ys) - 1))
        return float(ys[idx])
    return float(raw)


def score(card, features_dict):
    x = np.array([float(features_dict.get(name, card["mean"][i]))
                  for i, name in enumerate(card["features"])])
    z = (x - np.array(card["mean"])) / np.array(card["std"])
    raw = float(1.0 / (1.0 + np.exp(-(z @ np.array(card["weights"]) + card["bias"]))))
    p = apply_calibration(card, raw)
    return {"readmission_risk": round(p, 4),
            "threshold": card["threshold"],
            "flag_for_follow_up": bool(p >= card["threshold"])}


if __name__ == "__main__":
    card = load_card()
    payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(score(card, payload), indent=2))
