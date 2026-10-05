"""
Sparse Mixture-of-Experts (MoE) token routing, implemented from scratch in NumPy.

This is the core efficiency idea behind sparse MoE language models such as
Reflection AI's Beam (open-weight release, 5 Oct 2026): a 501B-parameter model
where each token is routed to only a small subset of "expert" feed-forward
networks, so roughly 23B parameters are active per token (about 4.6 percent).
Sparse activation is why these models can claim 3-4x less inference compute
than a dense model of the same total size.

What this module implements:
  1. TopKGate   - a learned router: softmax over experts, keep top-k, renormalise.
  2. Expert     - a small two-layer feed-forward network (ReLU MLP).
  3. MoELayer   - routes every token through its top-k experts, weighted sum out.
  4. load_balance_loss - the Switch Transformer auxiliary loss that keeps
     routing balanced so no expert is starved or overloaded.
  5. flop comparison helpers - dense vs sparse compute, plus a Beam-scale
     projection using Reflection's published numbers.

Manual backprop is used throughout (no autograd dependency), with the top-k
selection treated as fixed during the backward pass, which is standard practice
(GShard / Switch Transformer).
"""

import numpy as np


def softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class TopKGate:
    """Learned token-to-expert router with top-k sparse selection."""

    def __init__(self, d_model, n_experts, k=2, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / np.sqrt(d_model), size=(d_model, n_experts))
        self.b = np.zeros(n_experts)
        self.n_experts = n_experts
        self.k = k

    def forward(self, x):
        """Route tokens.

        Returns:
            gate : (T, E) sparse gate weights (Switch-Transformer style: the raw
                   softmax probability of each selected expert, so the router
                   keeps a learning signal even with k=1)
            idx  : (T, k) indices of the selected experts per token
        """
        logits = x @ self.W + self.b
        probs = softmax(logits, axis=1)
        idx = np.argsort(-probs, axis=1)[:, : self.k]
        rows = np.arange(x.shape[0])[:, None]
        gate = np.zeros_like(probs)
        gate[rows, idx] = probs[rows, idx]
        cache = (x, logits, probs, idx)
        return gate, idx, cache

    def backward(self, dgate, cache):
        """Backprop through the top-k masked softmax (selection held fixed)."""
        x, logits, probs, idx = cache
        rows = np.arange(x.shape[0])[:, None]
        dprobs = np.zeros_like(probs)
        dprobs[rows, idx] = dgate[rows, idx]
        # standard softmax jacobian
        dlogits = probs * (dprobs - (dprobs * probs).sum(axis=1, keepdims=True))
        dW = x.T @ dlogits
        db = dlogits.sum(axis=0)
        return dW, db


class Expert:
    """One feed-forward expert: Linear -> ReLU -> Linear."""

    def __init__(self, d_model, d_ff, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / d_model), size=(d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / d_ff), size=(d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out, (x, h)

    def backward(self, dout, cache):
        x, h = cache
        dW2 = h.T @ dout
        db2 = dout.sum(axis=0)
        dh = dout @ self.W2.T
        dh[h <= 0.0] = 0.0
        dW1 = x.T @ dh
        db1 = dh.sum(axis=0)
        dx = dh @ self.W1.T
        return dx, (dW1, db1, dW2, db2)

    def n_params(self):
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)


class MoELayer:
    """Sparse MoE layer: route each token to its top-k experts, weighted sum."""

    def __init__(self, d_model, d_ff, n_experts, k=2, seed=0):
        self.gate = TopKGate(d_model, n_experts, k=k, seed=seed)
        self.experts = [Expert(d_model, d_ff, seed=seed + 1 + i) for i in range(n_experts)]
        self.n_experts = n_experts
        self.k = k
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x):
        gate, idx, gate_cache = self.gate.forward(x)
        T = x.shape[0]
        out = np.zeros_like(x)
        expert_caches = []
        rows = np.arange(T)[:, None]
        for e, expert in enumerate(self.experts):
            mask = idx == e                      # (T, k) where expert e was chosen
            tok = np.where(mask.any(axis=1))[0]  # tokens routed to e
            if tok.size == 0:
                expert_caches.append(None)
                continue
            eo, ecache = expert.forward(x[tok])
            w = gate[tok, e].reshape(-1, 1)
            out[tok] += w * eo
            expert_caches.append((tok, mask[tok], ecache))
        cache = (x, gate, idx, gate_cache, expert_caches)
        return out, cache

    def train_step(self, x, target, lr=1e-2, aux_coef=1e-2, train_experts=True,
                   train_gate=True):
        """One supervised step: MSE to target plus load-balancing auxiliary loss.

        train_experts / train_gate allow freezing either side, e.g. to train
        the router against fixed, already-specialised experts.
        """
        out, cache = self.forward(x)
        x_in, gate, idx, gate_cache, expert_caches = cache
        T = x.shape[0]

        # main loss: mean squared error
        diff = out - target
        mse = float((diff ** 2).mean())
        dout = 2.0 * diff / diff.size

        # auxiliary load-balancing loss (Switch Transformer style):
        #   aux = E * sum_i( f_i * P_i )
        # f_i = fraction of tokens routed to expert i, P_i = mean gate prob.
        probs = softmax(gate_cache[1], axis=1)
        counts = np.bincount(idx.ravel(), minlength=self.n_experts)
        f = counts / T
        P = probs.mean(axis=0)
        aux = self.n_experts * float((f * P).sum())

        # backprop through experts (gate selection held fixed)
        dgate = np.zeros_like(gate)
        rows = np.arange(T)[:, None]
        for e, expert in enumerate(self.experts):
            ec = expert_caches[e]
            if ec is None:
                continue
            tok, emask, xcache = ec
            eo, _ = expert.forward(x[tok])
            w = gate[tok, e].reshape(-1, 1)
            d_eo = dout[tok] * w
            dx_e, grads = expert.backward(d_eo, xcache)
            dW1, db1, dW2, db2 = grads
            if train_experts:
                expert.W1 -= lr * dW1
                expert.b1 -= lr * db1
                expert.W2 -= lr * dW2
                expert.b2 -= lr * db2
            # gradient wrt the gate weight of expert e on these tokens
            dgate[tok, e] = (dout[tok] * eo).sum(axis=1)

        # aux loss gradient flows only through P (mean gate probs), f is constant
        dP = aux_coef * self.n_experts * f / T
        dlogits_aux = probs * (dP - (dP * probs).sum())
        dW_aux = x_in.T @ dlogits_aux
        db_aux = dlogits_aux.sum(axis=0)

        dW_gate, db_gate = self.gate.backward(dgate, gate_cache)
        if train_gate:
            self.gate.W -= lr * (dW_gate + dW_aux)
            self.gate.b -= lr * (db_gate + db_aux)
        return mse, aux

    # -- bookkeeping -----------------------------------------------------
    def expert_token_counts(self, idx):
        return np.bincount(idx.ravel(), minlength=self.n_experts)

    def total_params(self):
        return sum(e.n_params() for e in self.experts) + int(self.gate.W.size)

    def active_params_per_token(self):
        per_expert = self.experts[0].n_params()
        return self.k * per_expert + int(self.gate.W.size)


def load_balance_loss(f, P, n_experts):
    """Switch Transformer auxiliary loss from routing stats (pure function)."""
    return float(n_experts * (f * P).sum())


def flop_report(d_model, d_ff, n_experts, k, n_tokens):
    """Compare dense vs sparse FFN FLOPs for n_tokens tokens (forward pass)."""
    per_expert = 2 * 2 * d_model * d_ff          # two matmuls, mul+add each
    dense = n_tokens * n_experts * per_expert
    sparse = n_tokens * k * per_expert
    return {
        "dense_flops": dense,
        "sparse_flops": sparse,
        "active_fraction": k / n_experts,
        "speedup": dense / sparse,
    }


def route_with_capacity(gate, idx, capacity_factor=1.25):
    """Drop tokens beyond per-expert capacity (GShard style).

    Returns per-token output mask (1.0 kept, 0.0 dropped) and drop fraction.
    """
    T, E = gate.shape
    k = idx.shape[1]
    capacity = int(np.ceil(capacity_factor * T * k / E))
    keep = np.ones(T)
    for e in range(E):
        assigned = np.where(idx == e)[0]
        if assigned.size > capacity:
            keep[assigned[capacity:]] = 0.0
    return keep, float(1.0 - keep.mean())
