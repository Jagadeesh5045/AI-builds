"""Probability calibration: Platt scaling and isotonic regression.

Both calibrators are fitted on the validation split only, never on the
test split they are later judged on.
"""

import numpy as np


class PlattScaler:
    """Fit a one-feature logistic regression on the raw score's logit."""

    def __init__(self, lr=0.2, epochs=600):
        self.lr = lr
        self.epochs = epochs
        self.a_ = 1.0
        self.b_ = 0.0

    @staticmethod
    def _logit(p, eps=1e-9):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def fit(self, p, y):
        z = self._logit(np.asarray(p, dtype=float))
        yf = np.asarray(y, dtype=float)
        a, b = 1.0, 0.0
        for _ in range(self.epochs):
            q = 1.0 / (1.0 + np.exp(-(a * z + b)))
            err = q - yf
            a -= self.lr * float(np.mean(err * z))
            b -= self.lr * float(np.mean(err))
        self.a_, self.b_ = float(a), float(b)
        return self

    def predict(self, p):
        z = self._logit(np.asarray(p, dtype=float))
        return 1.0 / (1.0 + np.exp(-(self.a_ * z + self.b_)))


class IsotonicCalibrator:
    """Isotonic regression via the pool-adjacent-violators algorithm."""

    def __init__(self):
        self.x_ = None
        self.y_ = None

    def fit(self, p, y):
        order = np.argsort(p, kind="mergesort")
        xs = np.asarray(p, dtype=float)[order]
        ys = np.asarray(y, dtype=float)[order].copy()
        w = np.ones_like(ys)
        # PAVA: merge adjacent blocks while monotonicity is violated
        blocks = [[i] for i in range(len(ys))]
        vals = list(ys)
        weights = list(w)
        i = 0
        merged_x = [[x] for x in xs]
        while i < len(vals) - 1:
            if vals[i] > vals[i + 1]:
                total_w = weights[i] + weights[i + 1]
                vals[i] = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / total_w
                weights[i] = total_w
                blocks[i] = blocks[i] + blocks[i + 1]
                merged_x[i] = merged_x[i] + merged_x[i + 1]
                del vals[i + 1], weights[i + 1], blocks[i + 1], merged_x[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1
        self.x_ = np.array([float(np.mean(mx)) for mx in merged_x])
        self.y_ = np.array(vals, dtype=float)
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=float)
        idx = np.searchsorted(self.x_, p, side="left")
        idx = np.clip(idx, 0, len(self.y_) - 1)
        return self.y_[idx]
