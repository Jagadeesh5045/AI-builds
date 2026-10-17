"""Train a selective SSM vs causal attention on selective copy; benchmark scaling.

Run:  python train_compare.py
Prints: training curves, held-out accuracy, a selectivity probe (does Delta
spike on markers/needles?), and forward+backward wall-time scaling that shows
the SSM's O(L) vs attention's O(L^2) growth -- the reason hybrid Mamba+attention
stacks (e.g. NVIDIA Nemotron 3.5) use SSM layers for cheap long-range
filtering.
"""

import time
import numpy as np
from ssm import SelectiveSSM, Adam, cross_entropy_and_grad
from attention import CausalAttention
from selective_copy import make_batch, VOCAB, N_DATA, MARKERS

D_MODEL = 32
D_STATE = 16
BATCH = 32
DATA_LEN = 48
STEPS = 800
LR = 5e-3


class Sequential:
    """Stack of sequence layers with the same forward/backward interface."""

    def __init__(self, layers):
        self.layers = layers
        self.params = [p for l in layers for p in l.params]

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dout):
        grads = []
        for l in reversed(self.layers):
            dout, g = l.backward(dout)
            grads = g + grads
        return dout, grads


class ToyLM:
    """Embedding + one sequence layer + vocab head, query-position loss."""

    def __init__(self, layer, rng):
        self.layer = layer
        self.emb = rng.normal(0, 0.3, size=(VOCAB, D_MODEL))
        self.head_W = rng.normal(0, 0.3, size=(D_MODEL, VOCAB))
        self.head_b = np.zeros(VOCAB)
        self.params = [self.emb, self.head_W, self.head_b] + layer.params
        self.opt = Adam(self.params, lr=LR)

    def forward(self, ids, qpos):
        B = ids.shape[0]
        x = self.emb[ids]
        h = self.layer.forward(x)
        q = h[np.arange(B)[:, None], qpos]
        return q @ self.head_W + self.head_b, (x, h, q)

    def step(self, ids, qpos, targets):
        B = ids.shape[0]
        logits, (x, h, q) = self.forward(ids, qpos)
        loss, dlogits = cross_entropy_and_grad(logits, targets)
        dq = dlogits @ self.head_W.T
        dW = np.einsum("bqd,bqv->dv", q, dlogits)
        db = dlogits.sum(axis=(0, 1))
        dh = np.zeros_like(h)
        dh[np.arange(B)[:, None], qpos] = dq
        dx, grads = self.layer.backward(dh)
        demb = np.zeros_like(self.emb)
        np.add.at(demb, ids, dx)
        self.opt.step([demb, dW, db] + grads)
        return loss


def evaluate(model, n_batches=16):
    rng = np.random.default_rng(999)
    correct = total = 0
    for _ in range(n_batches):
        ids, qpos, targets = make_batch(32, DATA_LEN, rng)
        logits, _ = model.forward(ids, qpos)
        pred = logits.argmax(axis=-1)
        correct += (pred == targets).sum()
        total += targets.size
    return correct / total


def main():
    rng = np.random.default_rng(42)
    ssm_model = ToyLM(SelectiveSSM(D_MODEL, D_STATE, np.random.default_rng(1)), rng)
    attn_model = ToyLM(CausalAttention(D_MODEL, np.random.default_rng(2)), rng)

    results = {}
    models = [
        ("ssm-1L", ToyLM(SelectiveSSM(D_MODEL, D_STATE, np.random.default_rng(1)), rng)),
        ("attn-1L", ToyLM(CausalAttention(D_MODEL, np.random.default_rng(2)), rng)),
        ("attn-2L", ToyLM(Sequential([CausalAttention(D_MODEL, np.random.default_rng(3)),
                                      CausalAttention(D_MODEL, np.random.default_rng(4))]), rng)),
    ]
    for name, model in models:
        print(f"\n=== training {name} ===", flush=True)
        t0 = time.time()
        data_rng = np.random.default_rng(123)  # same data order for both
        for step in range(STEPS):
            ids, qpos, targets = make_batch(BATCH, DATA_LEN, data_rng)
            loss = model.step(ids, qpos, targets)
            if step % 50 == 0 or step == STEPS - 1:
                print(f"  step {step:3d}  loss {loss:.4f}", flush=True)
        wall = time.time() - t0
        acc = evaluate(model)
        results[name] = (loss, acc, wall)
        print(f"  -> final loss {loss:.4f}, held-out acc {acc:.3f}, "
              f"train wall time {wall:.1f}s")

    # -- selectivity probe: does the SSM's step size spike on salient tokens?
    print("\n=== selectivity probe (mean Delta by token type, ssm-1L) ===")
    rng = np.random.default_rng(2026)
    d_noise, d_mark, d_need, d_query = [], [], [], []
    ssm1 = models[0][1]
    layer = ssm1.layer
    for _ in range(20):
        ids, qpos, targets = make_batch(8, DATA_LEN, rng)
        x = ssm1.emb[ids]
        layer.forward(x)
        Delta = layer.cache["Delta"]  # (B, L)
        B, L = Delta.shape
        for b in range(B):
            seq = ids[b]
            for t in range(DATA_LEN):
                tok = int(seq[t])
                if tok in MARKERS:
                    d_mark.append(Delta[b, t])
                elif t > 0 and int(seq[t - 1]) in MARKERS:
                    d_need.append(Delta[b, t])
                else:
                    d_noise.append(Delta[b, t])
            for t in range(DATA_LEN, L):
                d_query.append(Delta[b, t])
    print(f"  noise tokens : {np.mean(d_noise):.4f}")
    print(f"  markers      : {np.mean(d_mark):.4f}")
    print(f"  needles      : {np.mean(d_need):.4f}")
    print(f"  query tokens : {np.mean(d_query):.4f}")

    # -- scaling benchmark: forward+backward wall time vs sequence length
    print("\n=== scaling benchmark (forward+backward, batch=2, D=64, N=32) ===")
    print(f"  {'L':>6} | {'SSM (ms)':>9} | {'attention (ms)':>14} | ratio")
    ssm_b = SelectiveSSM(64, 32, np.random.default_rng(3))
    attn_b = CausalAttention(64, np.random.default_rng(4))
    prev = {}
    for L in [256, 512, 1024, 2048]:
        row = {}
        for nm, lyr in [("ssm", ssm_b), ("attn", attn_b)]:
            x = np.random.default_rng(5).normal(0, 0.5, size=(2, L, 64))
            dout = np.random.default_rng(6).normal(0, 0.5, size=(2, L, 64))
            lyr.forward(x)  # warmup
            t0 = time.time()
            lyr.forward(x)
            lyr.backward(dout)
            row[nm] = (time.time() - t0) * 1000
        ratio = row["attn"] / row["ssm"]
        growth = ""
        if prev:
            growth = (f"  [2x L: SSM x{row['ssm']/prev['ssm']:.2f}, "
                      f"attn x{row['attn']/prev['attn']:.2f}]")
        print(f"  {L:>6} | {row['ssm']:>9.1f} | {row['attn']:>14.1f} | "
              f"{ratio:>5.2f}x{growth}")
        prev = row

    print("\ntheory: SSM scan is O(L*D*N) with a fixed-size state; causal")
    print("attention materializes an LxL score matrix -> O(L^2*D) time and memory.")


if __name__ == "__main__":
    main()
