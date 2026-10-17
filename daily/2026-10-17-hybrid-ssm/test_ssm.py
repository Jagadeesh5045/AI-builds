"""Gradient checks (analytic vs finite differences) + smoke tests.

Run:  python test_ssm.py
"""

import numpy as np
from ssm import SelectiveSSM, cross_entropy_and_grad
from attention import CausalAttention
from selective_copy import make_batch, VOCAB


def rel_err(a, b):
    return float(np.abs(a - b).max() / (np.abs(a).max() + np.abs(b).max() + 1e-12))


def check_layer(layer_cls, d_model, d_state=None, name=""):
    rng = np.random.default_rng(0)
    layer = layer_cls(d_model, d_state, rng) if d_state else layer_cls(d_model, rng)
    x = rng.normal(0, 0.5, size=(2, 5, d_model))
    dout = rng.normal(0, 0.5, size=(2, 5, d_model))
    out = layer.forward(x)
    dx, grads = layer.backward(dout)
    assert out.shape == x.shape and dx.shape == x.shape

    eps = 1e-5
    # check a few parameter entries
    for pi in [0, -1]:
        p = layer.params[pi]
        idx = tuple(0 for _ in p.shape)
        old = p[idx]
        p[idx] = old + eps
        lp = (layer.forward(x) * dout).sum()
        p[idx] = old - eps
        lm = (layer.forward(x) * dout).sum()
        p[idx] = old
        num = (lp - lm) / (2 * eps)
        ana = grads[pi][idx]
        err = rel_err(np.array(num), np.array(ana))
        assert err < 1e-4, f"{name} param {pi}: rel err {err:.2e}"
    # check one input entry
    idx = (0, 2, 1)
    old = x[idx]
    x[idx] = old + eps
    lp = (layer.forward(x) * dout).sum()
    x[idx] = old - eps
    lm = (layer.forward(x) * dout).sum()
    num = (lp - lm) / (2 * eps)
    # recompute analytic dx with the ORIGINAL x
    x[idx] = old
    layer.forward(x)
    dx, _ = layer.backward(dout)
    err = rel_err(np.array(num), np.array(dx[idx]))
    assert err < 1e-4, f"{name} dx: rel err {err:.2e}"
    print(f"  {name}: gradient check OK")


def test_shapes_and_stability():
    rng = np.random.default_rng(1)
    ssm = SelectiveSSM(16, 8, rng)
    x = rng.normal(0, 1, size=(3, 64, 16))
    out = ssm.forward(x)
    assert np.all(np.isfinite(out)), "SSM forward produced non-finite values"
    # long sequence must not blow up: state decay keeps it bounded
    x2 = rng.normal(0, 1, size=(1, 2048, 16))
    out2 = ssm.forward(x2)
    assert np.all(np.isfinite(out2)) and np.abs(out2).max() < 1e4
    print("  shapes + long-sequence stability OK")


def test_smoke_training():
    """20 training steps on selective copy must reduce the loss."""
    from ssm import Adam
    rng = np.random.default_rng(7)
    D, N = 24, 8
    emb = rng.normal(0, 0.3, size=(VOCAB, D))
    head_W = rng.normal(0, 0.3, size=(D, VOCAB))
    head_b = np.zeros(VOCAB)
    layer = SelectiveSSM(D, N, np.random.default_rng(8))
    params = [emb, head_W, head_b] + layer.params
    opt = Adam(params, lr=5e-3)
    losses = []
    for step in range(20):
        ids, qpos, targets = make_batch(16, 40, rng)
        x = emb[ids]
        h = layer.forward(x)
        q = h[np.arange(16)[:, None], qpos]
        logits = q @ head_W + head_b
        loss, dlogits = cross_entropy_and_grad(logits, targets)
        losses.append(loss)
        dq = dlogits @ head_W.T
        dW = np.einsum("bqd,bqv->dv", q, dlogits)
        db = dlogits.sum(axis=(0, 1))
        dh = np.zeros_like(h)
        dh[np.arange(16)[:, None], qpos] = dq
        dx, grads = layer.backward(dh)
        demb = np.zeros_like(emb)
        np.add.at(demb, ids, dx)
        opt.step([demb, dW, db] + grads)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    print(f"  smoke training OK (loss {losses[0]:.3f} -> {losses[-1]:.3f})")


if __name__ == "__main__":
    print("gradient checks:")
    check_layer(SelectiveSSM, 8, 4, "SelectiveSSM")
    check_layer(CausalAttention, 8, None, "CausalAttention")
    test_shapes_and_stability()
    test_smoke_training()
    print("ALL TESTS PASSED")
