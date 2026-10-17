"""Causal single-head self-attention, from scratch in NumPy.

This is the baseline the selective SSM is compared against in the hybrid
story: NVIDIA's Nemotron 3.5 pairs Mamba-2 layers with attention layers in one
hybrid stack (Mamba for cheap long-range filtering, attention for precise
retrieval). Exact forward + reverse-mode backprop, verified against finite
differences in test_ssm.py.
"""

import numpy as np
from ssm import softmax


class CausalAttention:
    """Single-head causal attention with output projection + residual."""

    def __init__(self, d_model, rng):
        self.D = d_model
        s = 1.0 / np.sqrt(d_model)
        self.W_q = rng.normal(0, s, size=(d_model, d_model))
        self.W_k = rng.normal(0, s, size=(d_model, d_model))
        self.W_v = rng.normal(0, s, size=(d_model, d_model))
        self.W_o = rng.normal(0, s, size=(d_model, d_model))
        self.b_o = np.zeros(d_model)
        self.params = [self.W_q, self.W_k, self.W_v, self.W_o, self.b_o]

    def forward(self, x):
        B, L, D = x.shape
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(D)          # (B, L, L)
        mask = np.triu(np.ones((L, L)), k=1).astype(bool)
        scores = np.where(mask[None], -1e9, scores)
        p = softmax(scores, axis=-1)                            # (B, L, L)
        y = p @ v                                               # (B, L, D)
        out = y @ self.W_o + self.b_o + x
        self.cache = dict(x=x, q=q, k=k, v=v, p=p, y=y)
        return out

    def backward(self, dout):
        c = self.cache
        x, q, k, v, p, y = c["x"], c["q"], c["k"], c["v"], c["p"], c["y"]
        B, L, D = x.shape

        dy = dout @ self.W_o.T
        dW_o = np.einsum("bld,blm->dm", y, dout)
        db_o = dout.sum(axis=(0, 1))
        dx = dout.copy()

        # y = p @ v
        dp = dy @ v.transpose(0, 2, 1)                          # (B, L, L)
        dv = p.transpose(0, 2, 1) @ dy                          # (B, L, D)

        # softmax rows: dp_logits = p * (dp - sum(dp*p))
        dscores = p * (dp - (dp * p).sum(axis=-1, keepdims=True))
        mask = np.triu(np.ones((L, L)), k=1).astype(bool)
        dscores = np.where(mask[None], 0.0, dscores)
        dscores = dscores / np.sqrt(D)

        dq = dscores @ k
        dk = dscores.transpose(0, 2, 1) @ q
        dW_q = np.einsum("bld,blm->dm", x, dq)
        dW_k = np.einsum("bld,blm->dm", x, dk)
        dW_v = np.einsum("bld,blm->dm", x, dv)
        dx = dx + dq @ self.W_q.T + dk @ self.W_k.T + dv @ self.W_v.T
        return dx, [dW_q, dW_k, dW_v, dW_o, db_o]
