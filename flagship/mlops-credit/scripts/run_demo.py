"""Single entry point: train -> canary -> drift, end to end.

Run from the project root:  python scripts/run_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from train import train  # noqa: E402
from canary import canary  # noqa: E402
from drift import drift  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("STEP 1: TRAIN AND REGISTER")
    print("=" * 60)
    t = train()
    print()
    print("=" * 60)
    print("STEP 2: CANARY DEPLOY (90% champion / 10% challenger)")
    print("=" * 60)
    c = canary()
    print()
    print("=" * 60)
    print("STEP 3: DRIFT DETECTION AND RETRAINING")
    print("=" * 60)
    d = drift()
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, m in t["metrics"].items():
        print(f"  {name}: ROC-AUC {m['roc_auc']}, accuracy {m['accuracy']}, "
              f"F1 {m['f1']} (test n={t['test_size']})")
    print(f"  Champion: {t['champion']} | Challenger: {t['challenger']}")
    print(f"  Canary: champion ROC-AUC {c['champion_roc_auc']} vs "
          f"challenger {c['challenger_roc_auc']} -> {c['decision']}")
    print(f"  Drift: PSI {d['psi']} -> {d['decision']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
