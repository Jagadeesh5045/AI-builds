"""
Toy masked-diffusion language model, in the style of Inception's Mercury 3
(released 19 Oct 2026): instead of writing left-to-right like an
autoregressive model, a diffusion LM starts from an all-[MASK] draft and
refines every position *in parallel* over a few denoising steps.

Absorbing-state (mask) diffusion:
  forward  q(x_t | x_0): each token is independently replaced by [MASK]
           with probability t  (t = noise level in [0, 1])
  reverse  p(x_0 | x_t, t): a small denoiser predicts the original tokens
           from the masked sequence, conditioned on the noise level t.

Sampling ("confidence-based decoding", like LLaDA / Mercury): start from
all masks; each step unmask the most confident predictions, keep the rest
masked, repeat. A length-L sentence is produced in S << L forward passes.

Everything is numpy + stdlib so it runs anywhere. Not affiliated with
Inception; Mercury figures in README are Inception's own claims.
"""

import numpy as np

# --------------------------------------------------------------------------
# Toy grammar with CROSS-TOKEN dependencies (so context actually helps):
# two independent clauses of "DET SUBJ VERB PREP DET OBJ", e.g.
# "the cat sat on the mat a bird flew near a tray"  (L = 12 tokens).
# The verb depends on the subject (bird->flew, fish->swam, cat/dog->sat|ran),
# so a masked verb can only be recovered by looking at the subject position.
# A denoiser without cross-position mixing cannot beat marginal guessing.
# --------------------------------------------------------------------------
DET = ["the", "a"]
SUBJ = ["cat", "dog", "bird", "fish"]
VERBS = ["sat", "ran", "flew", "swam"]
VERB_MAP = {"cat": ["sat"], "dog": ["ran"],
            "bird": ["flew"], "fish": ["swam"]}
PREP = ["on", "near", "under"]
OBJ = ["mat", "tray"]

WORDS = DET + SUBJ + VERBS + PREP + OBJ
V = len(WORDS)
WORD2ID = {w: i for i, w in enumerate(WORDS)}
ID2WORD = {i: w for w, i in WORD2ID.items()}
MASK = V  # extra id for the absorbing mask token
L = 12


def sample_sentences(n, rng):
    """Draw n grammatical token-id sequences, shape (n, L)."""
    out = np.zeros((n, L), dtype=np.int64)
    for b in range(n):
        ids = []
        for _ in range(2):  # two clauses
            subj = rng.choice(SUBJ)
            ids += [WORD2ID[rng.choice(DET)],
                    WORD2ID[subj],
                    WORD2ID[rng.choice(VERB_MAP[subj])],
                    WORD2ID[rng.choice(PREP)],
                    WORD2ID[rng.choice(DET)],
                    WORD2ID[rng.choice(OBJ)]]
        out[b] = ids
    return out


def is_grammatical(ids):
    """Check subject-verb agreement across both clauses."""
    for c in range(2):
        seg = [ID2WORD[int(i)] for i in ids[6 * c:6 * c + 6]]
        if seg[0] not in DET or seg[1] not in SUBJ or seg[3] not in PREP:
            return False
        if seg[2] not in VERB_MAP.get(seg[1], []):
            return False
        if seg[4] not in DET or seg[5] not in OBJ:
            return False
    return True


def words_of(ids):
    return " ".join(ID2WORD[int(i)] if int(i) != MASK else "[MASK]" for i in ids)


# --------------------------------------------------------------------------
# Forward (noising) process
# --------------------------------------------------------------------------
def forward_process(x0, t, rng):
    """Mask each token independently with probability t.

    x0: (B, L) clean ids; t: (B,) noise levels. Returns (xt, was_masked).
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    was_masked = rng.random(x0.shape) < t
    xt = np.where(was_masked, MASK, x0)
    return xt, was_masked


# --------------------------------------------------------------------------
# Denoiser: token+position+noise-level embeddings -> MLP -> per-position logits
#
# Cross-position mixing is the whole point of a diffusion LM (every output
# position may depend on every input position, unlike a causal AR model).
# Here it is one single-head self-attention layer over the *observed*
# (unmasked) tokens: each position queries the visible tokens and pulls a
# context vector from them. A masked verb position can therefore learn to
# attend to the subject position and recover subject->verb agreement.
# --------------------------------------------------------------------------
class MaskedDiffuser:
    def __init__(self, d=32, h=64, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.5, (V + 1, d))      # token embeddings (+MASK)
        self.P = rng.normal(0, 0.5, (L, d))         # positional embeddings
        self.wt = rng.normal(0, 0.5, d)             # noise-level conditioning
        sc = np.sqrt(d)
        self.Wq = rng.normal(0, 1.0 / sc, (d, d))   # attention Q/K/V
        self.Wk = rng.normal(0, 1.0 / sc, (d, d))
        self.Wv = rng.normal(0, 1.0 / sc, (d, d))
        self.W1 = rng.normal(0, np.sqrt(2.0 / d), (d, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, np.sqrt(2.0 / h), (h, d))
        self.b2 = np.zeros(d)
        self.Wo = rng.normal(0, np.sqrt(2.0 / d), (d, V))
        self.bo = np.zeros(V)
        # Adam state, one slot per parameter
        self._m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self._step = 0

    def params(self):
        return {"E": self.E, "P": self.P, "wt": self.wt,
                "Wq": self.Wq, "Wk": self.Wk, "Wv": self.Wv,
                "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                "Wo": self.Wo, "bo": self.bo}

    def forward(self, x, t):
        """x: (B, L) ids incl. MASK; t: (B,) noise levels -> (B, L, V) logits."""
        B = x.shape[0]
        t = np.asarray(t, dtype=np.float64).reshape(B, 1, 1)
        tok = self.E[x]                                   # (B, L, d)
        tp = tok + self.P[None, :, :]                   # positions BEFORE attn
        # single-head self-attention over observed tokens only.
        # Queries carry their position, so a masked verb slot learns to
        # attend to the subject slot (unlike a position-blind query, which
        # produces the same context for every masked position).
        q = tp @ self.Wq
        k = tp @ self.Wk
        v = tp @ self.Wv
        s = q @ k.transpose(0, 2, 1) / np.sqrt(tp.shape[-1])
        key_ok = (x != MASK).astype(np.float64)[:, None, :]   # (B, 1, L)
        s = s + (1.0 - key_ok) * -1e9
        s = s - s.max(axis=-1, keepdims=True)
        attn = np.exp(s)
        attn = attn / attn.sum(axis=-1, keepdims=True)    # (B, L, L)
        ctx = attn @ v                                    # (B, L, d)
        emb = tp + t * self.wt[None, None, :] + ctx
        z1 = emb @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        res = emb + z2                      # residual around the MLP block
        logits = res @ self.Wo + self.bo
        cache = {"x": x, "t": t, "tp": tp, "q": q, "k": k, "v": v,
                 "attn": attn,
                 "emb": emb, "z1": z1, "a1": a1, "z2": z2, "res": res}
        return logits, cache

    @staticmethod
    def loss_and_grad_logits(logits, targets, was_masked):
        """Cross-entropy over masked positions only; returns (loss, dlogits)."""
        B, Ll, _ = logits.shape
        m = logits - logits.max(axis=-1, keepdims=True)
        logp = m - np.log(np.exp(m).sum(axis=-1, keepdims=True))
        nll = -logp[np.arange(B)[:, None], np.arange(Ll)[None, :], targets]
        nll = nll * was_masked
        denom = max(was_masked.sum(), 1)
        loss = nll.sum() / denom
        dlogits = np.exp(logp)
        b_idx, l_idx = np.nonzero(was_masked)
        dlogits[b_idx, l_idx, targets[b_idx, l_idx]] -= 1.0
        dlogits *= (was_masked / denom)[..., None]
        return loss, dlogits

    def backward(self, dlogits, cache):
        x, t = cache["x"], cache["t"]
        tp, q, k, v, attn = (cache[n] for n in
                             ("tp", "q", "k", "v", "attn"))
        emb, z1, a1, z2, res = (cache[n] for n in
                                ("emb", "z1", "a1", "z2", "res"))
        d = tp.shape[-1]
        dres = dlogits @ self.Wo.T
        dWo = res.reshape(-1, res.shape[-1]).T @ dlogits.reshape(-1, V)
        dbo = dlogits.sum(axis=(0, 1))
        dz2 = dres                      # residual branch
        demb_res = dres                 # skip branch
        dW2 = a1.reshape(-1, a1.shape[-1]).T @ dz2.reshape(-1, dz2.shape[-1])
        db2 = dz2.sum(axis=(0, 1))
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = emb.reshape(-1, emb.shape[-1]).T @ dz1.reshape(-1, dz1.shape[-1])
        db1 = dz1.sum(axis=(0, 1))
        demb = dz1 @ self.W1.T + demb_res
        # attention backward: ctx = attn @ v ; attn = softmax(q k^T / sqrt(d))
        dctx = demb
        dattn = dctx @ v.transpose(0, 2, 1)          # (B, L, L)
        dv = attn.transpose(0, 2, 1) @ dctx
        ds = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
        sc = np.sqrt(d)
        dq = ds @ k / sc
        dk = ds.transpose(0, 2, 1) @ q / sc
        dtp_attn = dq @ self.Wq.T + dk @ self.Wk.T + dv @ self.Wv.T
        dWq = np.einsum("bld,blD->dD", tp, dq)
        dWk = np.einsum("bld,blD->dD", tp, dk)
        dWv = np.einsum("bld,blD->dD", tp, dv)
        dtp = demb + dtp_attn
        # tp = tok + P : split the gradient
        dE = np.zeros_like(self.E)
        np.add.at(dE, x.ravel(), dtp.reshape(-1, d))
        dP = dtp.sum(axis=0)
        dwt = (demb * t).sum(axis=(0, 1))
        return {"E": dE, "P": dP, "wt": dwt,
                "Wq": dWq, "Wk": dWk, "Wv": dWv,
                "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                "Wo": dWo, "bo": dbo}

    def adam_step(self, grads, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._step += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            mh = self._m[k] / (1 - b1 ** self._step)
            vh = self._v[k] / (1 - b2 ** self._step)
            p = self.params()[k]
            p -= lr * mh / (np.sqrt(vh) + eps)

    def train(self, n_steps=1200, batch=256, seed=1, log_every=400):
        rng = np.random.default_rng(seed)
        losses = []
        for step in range(n_steps):
            x0 = sample_sentences(batch, rng)
            t = rng.uniform(0.05, 1.0, size=batch)
            xt, was_masked = forward_process(x0, t, rng)
            logits, cache = self.forward(xt, t)
            loss, dlogits = self.loss_and_grad_logits(logits, x0, was_masked)
            self.adam_step(self.backward(dlogits, cache))
            losses.append(loss)
            if log_every and (step + 1) % log_every == 0:
                print(f"  step {step + 1}/{n_steps}  loss {loss:.4f}", flush=True)
        return losses


# --------------------------------------------------------------------------
# Reverse (generative) process: confidence-based parallel decoding
# --------------------------------------------------------------------------
def decode(model, x_init, steps=4, seed=0, count_calls=None):
    """Fill [MASK] slots in parallel over `steps` refinement rounds.

    x_init: (L,) ids, MASK = to fill, anything else = fixed context.
    Each round unmasks the most confident predictions first.
    Returns the completed (L,) ids.
    """
    rng = np.random.default_rng(seed)
    x = np.array(x_init, dtype=np.int64).reshape(1, L)
    fixed = x != MASK
    for s in range(steps):
        n_masked = int((~fixed).sum())
        if n_masked == 0:
            break
        t = np.full(1, n_masked / L)          # noise level ~= masked fraction
        logits, _ = model.forward(x, t)
        if count_calls is not None:
            count_calls[0] += 1
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        conf = probs.max(-1)[0]
        pred = probs.argmax(-1)[0]
        # unmask the top-k most confident of the remaining slots
        k = max(1, int(np.ceil(n_masked / (steps - s))))
        open_idx = np.nonzero(~fixed[0])[0]
        # jitter ties so decoding is not pathologically deterministic
        order = open_idx[np.argsort(-conf[open_idx] + rng.random(len(open_idx)) * 1e-6)]
        take = order[:k]
        x[0, take] = pred[take]
        fixed[0, take] = True
    # safety net: argmax-fill anything still masked
    if (x == MASK).any():
        logits, _ = model.forward(x, np.zeros(1))
        x[x == MASK] = logits.argmax(-1)[x == MASK]
    return x[0]


def reconstruct_accuracy(model, t_level, n=400, seed=99):
    """Fraction of masked tokens recovered at a fixed noise level."""
    rng = np.random.default_rng(seed)
    x0 = sample_sentences(n, rng)
    xt, was_masked = forward_process(x0, np.full(n, t_level), rng)
    logits, _ = model.forward(xt, np.full(n, t_level))
    pred = logits.argmax(-1)
    ok = (pred == x0) & was_masked
    return ok.sum() / max(was_masked.sum(), 1)


def reconstruct_verb_accuracy(model, n=400, seed=99):
    """Mask ONLY the two verb slots; every subject stays visible.

    The verb is a deterministic function of the subject (bird->flew, ...),
    so this isolates cross-position reasoning: a position-blind denoiser
    scores ~0.25 (marginal), one that learned subject->verb agreement ~1.0.
    """
    rng = np.random.default_rng(seed)
    x0 = sample_sentences(n, rng)
    xt = x0.copy()
    verb_pos = np.array([2, 8])
    xt[:, verb_pos] = MASK
    logits, _ = model.forward(xt, np.full(n, len(verb_pos) / L))
    pred = logits.argmax(-1)
    return (pred[:, verb_pos] == x0[:, verb_pos]).mean()
