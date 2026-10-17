"""Selective state-space layer (Mamba-style), from scratch in NumPy.

Trending context: NVIDIA's Nemotron 3.5 Super VL (Oct 2026) ships a *hybrid*
Mamba-2 + attention + MoE architecture; Meta's Muse Spark Open (Oct 2026) is
another open-weight giant. This file implements the core idea that makes the
"selective" part work: the recurrence parameters (step size Delta, input/output
projections B and C) are *functions of the input*, so the layer can decide per
token what to remember and what to forget -- the mechanism behind Mamba's
selective-copy ability.

Discrete-time model (simplified Mamba / S6):
    h_t = Abar_t * h_{t-1} + Bbar_t * x_t        (elementwise, per channel/state)
    y_t = sum_n C_{t,n} * h_{t,d,n}
with zero-order-hold discretization
    Abar_t = exp(Delta_t * A),   Bbar_t = Delta_t * B_t
(A kept negative via A = -exp(log_A) so the recurrence is stable.)

Everything below is exact forward + reverse-mode backprop, verified against
finite differences in test_ssm.py. Shapes: B=batch, L=length, D=channels,
N=state dim.
"""

import numpy as np


def softplus(z):
    # numerically stable softplus
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy_and_grad(logits, targets):
    """logits: (B, Q, V), targets: (B, Q) int. Returns (mean loss, dlogits)."""
    B, Q, V = logits.shape
    probs = softmax(logits)
    nll = -np.log(probs[np.arange(B)[:, None], np.arange(Q)[None, :], targets] + 1e-12)
    loss = nll.mean()
    dlogits = probs
    dlogits[np.arange(B)[:, None], np.arange(Q)[None, :], targets] -= 1.0
    dlogits = dlogits / (B * Q)
    return loss, dlogits


class Adam:
    """Minimal Adam for a list of parameter arrays (in-place update)."""

    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class SelectiveSSM:
    """One selective state-space block with residual + output projection."""

    def __init__(self, d_model, d_state, rng):
        self.D, self.N = d_model, d_state
        self.log_A = rng.uniform(-3.0, 0.0, size=(d_model, d_state))
        self.W_B = rng.normal(0, 0.2, size=(d_model, d_state))
        self.b_B = np.zeros(d_state)
        self.W_C = rng.normal(0, 0.2, size=(d_model, d_state))
        self.b_C = np.zeros(d_state)
        self.w_d = rng.normal(0, 0.2, size=d_model)
        # Start with a SMALL step size: abar = exp(Delta*A) ~= 0.98, so the
        # state keeps ~40 steps of memory at init and gradients flow back.
        # (With b_d = 0 the state forgets in ~5 steps and nothing can learn.)
        self.b_d = np.full((), -2.5)
        self.W_o = rng.normal(0, 0.2, size=(d_model, d_model))
        self.b_o = np.zeros(d_model)
        self.params = [self.log_A, self.W_B, self.b_B, self.W_C, self.b_C,
                       self.w_d, self.b_d, self.W_o, self.b_o]

    # -- forward ---------------------------------------------------------
    def forward(self, x):
        B, L, D = x.shape
        N = self.N
        A = -np.exp(self.log_A)                                  # (D, N), negative
        Bt = x @ self.W_B + self.b_B                             # (B, L, N)
        Ct = x @ self.W_C + self.b_C                             # (B, L, N)
        s = x @ self.w_d + self.b_d                              # (B, L)
        Delta = softplus(s)                                      # (B, L)
        abar = np.exp(Delta[:, :, None, None] * A[None, None])   # (B, L, D, N)
        bbar = Delta[:, :, None, None] * Bt[:, :, None, :]       # (B, L, D, N)

        h = np.zeros((B, D, N))
        y = np.zeros((B, L, D))
        h_hist = np.zeros((B, L, D, N))
        for t in range(L):
            h = abar[:, t] * h + bbar[:, t] * x[:, t, :, None]
            h_hist[:, t] = h
            y[:, t] = (Ct[:, t, None, :] * h).sum(axis=-1)
        out = y @ self.W_o + self.b_o + x                        # residual

        self.cache = dict(x=x, A=A, Bt=Bt, Ct=Ct, s=s, Delta=Delta,
                          abar=abar, bbar=bbar, y=y, h_hist=h_hist)
        return out

    # -- backward --------------------------------------------------------
    def backward(self, dout):
        c = self.cache
        x, A, Bt, Ct, s = c["x"], c["A"], c["Bt"], c["Ct"], c["s"]
        Delta, abar, bbar, y, h_hist = (c["Delta"], c["abar"], c["bbar"],
                                        c["y"], c["h_hist"])
        B, L, D = x.shape
        N = self.N

        dy = dout @ self.W_o.T                                    # (B, L, D)
        dW_o = np.einsum("bld,blm->dm", y, dout)
        db_o = dout.sum(axis=(0, 1))
        dx = dout.copy()                                          # residual path

        dlog_A = np.zeros_like(self.log_A)
        dW_B = np.zeros_like(self.W_B); db_B = np.zeros_like(self.b_B)
        dW_C = np.zeros_like(self.W_C); db_C = np.zeros_like(self.b_C)
        dw_d = np.zeros_like(self.w_d); db_d = np.zeros_like(self.b_d)

        dh_next = np.zeros((B, D, N))   # dL/dh_t accumulated from the future
        for t in range(L - 1, -1, -1):
            h_t = h_hist[:, t]                                   # (B, D, N)
            h_prev = h_hist[:, t - 1] if t > 0 else np.zeros_like(h_t)
            # dL/dh_t : through y_t, plus through h_{t+1} via abar_{t+1}
            dh = Ct[:, t, None, :] * dy[:, t, :, None]
            if t < L - 1:
                dh = dh + abar[:, t + 1] * dh_next

            da = dh * h_prev                                     # dL/d abar_t
            dbbar = dh * x[:, t, :, None]                        # dL/d bbar_t
            dx[:, t] += (dh * bbar[:, t]).sum(axis=-1)            # via bbar*x_t
            dCt = (dy[:, t, :, None] * h_t).sum(axis=1)           # (B, N)

            # abar_t = exp(Delta_t * A)
            dA_term = da * abar[:, t]                            # dL/d(Delta*A)
            dDelta = (dA_term * A[None]).sum(axis=(1, 2))        # (B,)
            dlog_A += (dA_term * Delta[:, t, None, None]).sum(axis=0) * A

            # bbar_t = Delta_t * B_t  (Delta broadcast over D,N)
            dDelta = dDelta + (dbbar * Bt[:, t, None, :]).sum(axis=(1, 2))
            dBt = (dbbar * Delta[:, t, None, None]).sum(axis=1)   # (B, N)

            # Delta_t = softplus(s_t)
            ds = dDelta * sigmoid(s[:, t])                       # (B,)
            dw_d += np.einsum("b,bd->d", ds, x[:, t])
            db_d += ds.sum()
            dx[:, t] += ds[:, None] * self.w_d

            # B_t = x_t W_B + b_B ;  C_t = x_t W_C + b_C
            dW_B += x[:, t].T @ dBt
            db_B += dBt.sum(axis=0)
            dx[:, t] += dBt @ self.W_B.T
            dW_C += x[:, t].T @ dCt
            db_C += dCt.sum(axis=0)
            dx[:, t] += dCt @ self.W_C.T

            dh_next = dh

        grads = [dlog_A, dW_B, db_B, dW_C, db_C, dw_d, db_d, dW_o, db_o]
        return dx, grads
