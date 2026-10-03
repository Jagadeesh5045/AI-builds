"""Decision-head model for agentic systems, numpy only.

Inspired by the October 2026 wave of "decision models" (Cloudflare Clef /
Clef-flash, 1 Oct 2026): instead of generating text and parsing it back into
an action, a decision model outputs a probability distribution over a fixed
set of discrete actions in a single forward pass. This module shows how that
pattern can look on the classical-ML side:

    agent context vector -> DecisionHead -> P(action) distribution

Features
- Multinomial logistic regression trained with mini-batch gradient descent.
- Calibrated-style confidence gating: if the top probability is below a
  threshold the head returns ESCALATE (hand off to a human) rather than the
  argmax action. This mirrors how Clef is positioned: agents "defer to a
  human when needed".
- Reliability (calibration) diagnostics: binned accuracy vs confidence.
- Latency benchmarking for batches of decisions.

The labels in the bundled demo are synthetic, but the wiring (features ->
probabilities -> gated decision -> JSON response) is the real, production
pattern used to put a cheap decision head in front of an expensive LLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

ACTIONS = ("proceed", "gather_context", "escalate", "defer")

FEATURE_NAMES = (
    "risk_score",            # 0..1, how risky the proposed action is
    "context_completeness",  # 0..1, how much of the needed info is available
    "tools_available",       # 0..1, fraction of required tools that are up
    "budget_remaining",      # 0..1, fraction of the agent's budget left
    "task_ambiguity",        # 0..1, how underspecified the task is
    "prior_success_rate",    # 0..1, historical success on similar tasks
)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out


@dataclass
class TrainingResult:
    epochs: int
    final_loss: float
    train_accuracy: float


class DecisionHead:
    """A small, fast decision model: context -> probability per action.

    Parameters
    ----------
    n_features: dimensionality of the agent-context vector.
    confidence_threshold: gate value; when the top probability is below it,
        decide() routes to "escalate" (human hand-off) instead of the argmax.
    seed: random seed for reproducible training.
    """

    def __init__(
        self,
        n_features: int,
        confidence_threshold: float = 0.55,
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.confidence_threshold = confidence_threshold
        self.n_actions = len(ACTIONS)
        rng = np.random.default_rng(seed)
        # W: (features+1 incl. bias) x actions, small init for stable training
        self.W = rng.normal(0.0, 0.01, size=(n_features + 1, self.n_actions))

    # -- core maths ------------------------------------------------------
    def _with_bias(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability distribution over actions for each context row."""
        return softmax(self._with_bias(X) @ self.W)

    def _loss(self, proba: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-12
        return float(-np.log(proba[np.arange(len(y)), y] + eps).mean())

    # -- training ---------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 400,
        batch_size: int = 64,
        lr: float = 0.5,
        seed: int = 7,
    ) -> TrainingResult:
        rng = np.random.default_rng(seed)
        Xb = self._with_bias(X)
        n = len(X)
        loss = 0.0
        for _ in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                b = idx[start : start + batch_size]
                proba = softmax(Xb[b] @ self.W)
                grad = Xb[b].T @ (proba - one_hot(y[b], self.n_actions)) / len(b)
                self.W -= lr * grad
            loss = self._loss(softmax(Xb @ self.W), y)
        preds = self.predict_proba(X).argmax(axis=1)
        acc = float((preds == y).mean())
        return TrainingResult(epochs=epochs, final_loss=loss, train_accuracy=acc)

    # -- inference ----------------------------------------------------------
    def decide(self, context: np.ndarray) -> dict:
        """Return a decision payload for one agent-context vector.

        Payload shape mirrors the decision-model API idea: a probability per
        option in a single response, no text generation involved.
        """
        proba = self.predict_proba(np.atleast_2d(context))[0]
        order = np.argsort(-proba)
        top_idx = int(order[0])
        top_prob = float(proba[top_idx])
        gated = bool(top_prob < self.confidence_threshold)
        action = "escalate" if gated else ACTIONS[top_idx]
        return {
            "action": action,
            "confidence": round(top_prob, 4),
            "gated_to_human": gated,
            "probabilities": {
                ACTIONS[i]: round(float(proba[i]), 4) for i in order
            },
            "threshold": self.confidence_threshold,
        }

    # -- diagnostics ---------------------------------------------------------
    def reliability(self, X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
        """Binned calibration: per confidence bin, accuracy vs mean confidence.

        Returns rows of {bin, n, mean_confidence, accuracy}. Well-calibrated
        models keep mean_confidence close to accuracy in every populated bin.
        """
        proba = self.predict_proba(X)
        conf = proba.max(axis=1)
        preds = proba.argmax(axis=1)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        rows = []
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (conf > lo) & (conf <= hi) if b > 0 else (conf <= hi)
            n = int(mask.sum())
            if n == 0:
                continue
            rows.append(
                {
                    "bin": f"({lo:.2f}, {hi:.2f}]",
                    "n": n,
                    "mean_confidence": round(float(conf[mask].mean()), 4),
                    "accuracy": round(float((preds[mask] == y[mask]).mean()), 4),
                }
            )
        return rows

    def expected_calibration_error(
        self, X: np.ndarray, y: np.ndarray, n_bins: int = 10
    ) -> float:
        proba = self.predict_proba(X)
        conf = proba.max(axis=1)
        preds = proba.argmax(axis=1)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (conf > lo) & (conf <= hi) if b > 0 else (conf <= hi)
            n = int(mask.sum())
            if n == 0:
                continue
            gap = abs(float(conf[mask].mean()) - float((preds[mask] == y[mask]).mean()))
            ece += (n / len(X)) * gap
        return round(ece, 4)

    def benchmark(self, X: np.ndarray, repeats: int = 20) -> dict:
        """Median milliseconds per decision over repeated batches."""
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.predict_proba(X)
            times.append((time.perf_counter() - t0) * 1000.0)
        med = float(np.median(times))
        return {
            "batch_size": int(len(X)),
            "median_batch_ms": round(med, 3),
            "median_per_decision_ms": round(med / len(X), 4),
        }
