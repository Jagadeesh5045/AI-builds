"""Logistic regression trained with full-batch gradient descent (NumPy).

No scikit-learn dependency: the model is small, and writing the optimiser
out makes the training step auditable in a code review.
"""

import numpy as np


class LogisticModel:
    def __init__(self, l2=1.0, lr=0.15, epochs=900):
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.mean_ = None
        self.std_ = None
        self.w_ = None
        self.b_ = 0.0

    def _standardise(self, X):
        return (X - self.mean_) / self.std_

    def fit(self, X, y):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Z = self._standardise(X)
        n, d = Z.shape
        w = np.zeros(d)
        b = 0.0
        yf = y.astype(float)
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
            err = p - yf
            grad_w = Z.T @ err / n + (self.l2 / n) * w
            grad_b = err.mean()
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.w_ = w
        self.b_ = float(b)
        return self

    def predict_proba(self, X):
        Z = self._standardise(X)
        return 1.0 / (1.0 + np.exp(-(Z @ self.w_ + self.b_)))
