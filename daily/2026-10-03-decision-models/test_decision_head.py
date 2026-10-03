"""Tests for the decision head."""

import numpy as np

from data_gen import make_dataset
from decision_head import ACTIONS, DecisionHead, softmax


def _trained(seed: int = 42) -> tuple[DecisionHead, np.ndarray, np.ndarray]:
    X, y = make_dataset(n=1200, seed=seed)
    head = DecisionHead(n_features=X.shape[1], seed=seed)
    head.fit(X, y, epochs=150, batch_size=64, lr=0.5, seed=seed)
    return head, X, y


def test_softmax_sums_to_one() -> None:
    p = softmax(np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def test_training_reduces_loss() -> None:
    head, X, y = _trained()
    # A trained head should beat a random-guess baseline comfortably.
    assert head._loss(head.predict_proba(X), y) < float(np.log(len(ACTIONS)))


def test_probabilities_sum_to_one() -> None:
    head, X, _ = _trained()
    assert np.allclose(head.predict_proba(X).sum(axis=1), 1.0)


def test_reproducible_training() -> None:
    X, y = make_dataset(n=600, seed=3)
    h1 = DecisionHead(n_features=X.shape[1], seed=11)
    h2 = DecisionHead(n_features=X.shape[1], seed=11)
    h1.fit(X, y, epochs=50, seed=5)
    h2.fit(X, y, epochs=50, seed=5)
    assert np.allclose(h1.W, h2.W)


def test_gate_routes_low_confidence_to_human() -> None:
    head, X, _ = _trained()
    # Threshold 1.0 means nothing passes the gate: every decision escalates.
    head_strict = DecisionHead(n_features=X.shape[1], confidence_threshold=1.0)
    head_strict.W = head.W.copy()
    out = head_strict.decide(X[0])
    assert out["gated_to_human"] is True
    assert out["action"] == "escalate"

    # Threshold 0.0 means nothing is gated: action is the raw argmax.
    head_open = DecisionHead(n_features=X.shape[1], confidence_threshold=0.0)
    head_open.W = head.W.copy()
    out = head_open.decide(X[0])
    assert out["gated_to_human"] is False
    top = max(head_open.predict_proba(X[:1])[0])
    assert out["confidence"] == round(float(top), 4)


def test_payload_shape() -> None:
    head, X, _ = _trained()
    out = head.decide(X[0])
    assert set(out) == {"action", "confidence", "gated_to_human",
                        "probabilities", "threshold"}
    assert set(out["probabilities"]) == set(ACTIONS)
    assert out["action"] in ACTIONS


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
