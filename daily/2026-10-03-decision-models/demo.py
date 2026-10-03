"""Demo: train a decision head, inspect it, and put it to work.

Run:  python demo.py

Shows:
1. Training a DecisionHead on synthetic agent telemetry.
2. Calibration (reliability table + expected calibration error).
3. Example decisions with confidence gating.
4. Latency: thousands of decisions per second on CPU, the "39ms" idea in
   miniature, vs. an expensive text-generating LLM round trip.
"""

from __future__ import annotations

import json
import time

import numpy as np

from data_gen import example_contexts, make_dataset
from decision_head import ACTIONS, DecisionHead

SIMULATED_LLM_ROUND_TRIP_MS = 900.0  # conservative proxy for LLM decide+parse


def main() -> None:
    X, y = make_dataset(n=4000, seed=123)
    head = DecisionHead(n_features=X.shape[1], confidence_threshold=0.55)
    res = head.fit(X, y, epochs=400, batch_size=64, lr=0.5)
    print(f"Trained {res.epochs} epochs | final loss {res.final_loss:.4f} "
          f"| train accuracy {res.train_accuracy:.3f}")

    # Holdout sanity check
    Xh, yh = make_dataset(n=1000, seed=999)
    holdout_acc = float((head.predict_proba(Xh).argmax(axis=1) == yh).mean())
    print(f"Holdout accuracy: {holdout_acc:.3f}\n")

    print("Calibration (accuracy should track confidence per bin):")
    for row in head.reliability(Xh, yh):
        print(f"  {row['bin']:>12}  n={row['n']:<5} conf={row['mean_confidence']:.3f} "
              f"acc={row['accuracy']:.3f}")
    print(f"Expected calibration error: {head.expected_calibration_error(Xh, yh)}\n")

    print("Example decisions:")
    for name, ctx in example_contexts():
        out = head.decide(ctx)
        gated = "  [GATED TO HUMAN]" if out["gated_to_human"] else ""
        probs = ", ".join(f"{a}={p:.2f}" for a, p in out["probabilities"].items())
        print(f"  {name:<34} -> {out['action']:<14} conf={out['confidence']:.2f}{gated}")
        print(f"    probs: {probs}")

    print("\nOne decision payload (the API shape a decision model returns):")
    print(json.dumps(head.decide(example_contexts()[0][1]), indent=2))

    print("\nLatency (decision head vs. text-generating LLM decision):")
    bench = head.benchmark(np.vstack([c for _, c in example_contexts()] * 200))
    per_ms = bench["median_per_decision_ms"]
    speedup = SIMULATED_LLM_ROUND_TRIP_MS / max(per_ms, 1e-9)
    print(f"  Decision head: {per_ms:.4f} ms per decision "
          f"({bench['batch_size']} decisions in {bench['median_batch_ms']:.2f} ms)")
    print(f"  Simulated LLM round trip: {SIMULATED_LLM_ROUND_TRIP_MS:.1f} ms per decision")
    print(f"  -> ~{speedup:,.0f}x faster; no text generated, no parsing needed")


if __name__ == "__main__":
    main()
