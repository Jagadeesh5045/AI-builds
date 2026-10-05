"""Unit tests for moe.py. Run: python test_moe.py"""

import numpy as np

from moe import (
    MoELayer,
    TopKGate,
    Expert,
    flop_report,
    load_balance_loss,
    route_with_capacity,
    softmax,
)


def test_gate_is_sparse_and_matches_softmax():
    g = TopKGate(d_model=8, n_experts=6, k=2, seed=1)
    x = np.random.default_rng(0).normal(size=(16, 8))
    gate, idx, _ = g.forward(x)
    assert gate.shape == (16, 6)
    assert (np.count_nonzero(gate, axis=1) == 2).all(), "exactly top-k nonzeros per row"
    # selected positions must hold the raw softmax probabilities of the top-k
    from moe import softmax as _softmax
    probs = _softmax(x @ g.W + g.b, axis=1)
    rows = np.arange(16)[:, None]
    assert np.allclose(gate[rows, idx], probs[rows, idx])
    # and the selected experts must be the argmax-k of the full distribution
    assert (idx == np.argsort(-probs, axis=1)[:, :2]).all()


def test_moe_output_shape():
    moe = MoELayer(8, 16, n_experts=4, k=2, seed=2)
    x = np.random.default_rng(0).normal(size=(10, 8))
    out, _ = moe.forward(x)
    assert out.shape == x.shape


def test_experts_specialise_with_training():
    # joint from-scratch training: the two token types (linear vs sinusoidal
    # subtask, living in shifted embedding regions) must end up routed to
    # different experts, with no expert starved.
    rng = np.random.default_rng(7)
    T, D = 512, 16
    ttype = np.repeat([0, 1], T // 2)
    perm = rng.permutation(T)
    x = rng.normal(size=(T, D)) + ttype[:, None] * 1.5
    x, ttype = x[perm], ttype[perm]
    A = rng.normal(0, 0.5, size=(D, D))
    B = rng.normal(0, 0.5, size=(D, D))
    target = np.where(ttype[:, None] == 0, x @ A, np.sin(3.0 * x) @ B)
    moe = MoELayer(D, 32, n_experts=4, k=1, seed=42)
    for _ in range(1500):
        moe.train_step(x, target, lr=3e-2, aux_coef=0.0)
    _, idx, _ = moe.gate.forward(x)
    table = np.zeros((2, 4), dtype=int)
    for t in range(T):
        table[ttype[t], idx[t, 0]] += 1
    # each type concentrates on its own experts (top-2 cover > 90%)
    for t in range(2):
        top2 = np.sort(table[t])[-2:].sum() / table[t].sum()
        assert top2 > 0.9, f"type {t} did not specialise: {table[t]}"
    # ...and the two types prefer different experts
    assert table[0].argmax() != table[1].argmax(), \
        f"both types collapsed onto one expert: {table.tolist()}"
    # no expert starved
    counts = moe.expert_token_counts(idx)
    assert counts.min() > 0, f"dead expert: {counts.tolist()}"


def test_training_reduces_mse():
    rng = np.random.default_rng(4)
    T, D = 96, 12
    x = rng.normal(size=(T, D))
    target = x @ rng.normal(0, 0.5, size=(D, D))
    moe = MoELayer(D, 24, n_experts=4, k=2, seed=5)
    first, _ = moe.train_step(x, target, lr=3e-2, aux_coef=1e-2)
    last = first
    for _ in range(150):
        last, _ = moe.train_step(x, target, lr=3e-2, aux_coef=1e-2)
    assert last < first * 0.5, f"mse did not fall: {first:.4f} -> {last:.4f}"


def test_aux_loss_formula():
    f = np.array([0.5, 0.5])
    P = np.array([0.5, 0.5])
    assert abs(load_balance_loss(f, P, 2) - 1.0) < 1e-9
    # perfectly imbalanced routing is penalised more than balanced routing
    bad = load_balance_loss(np.array([1.0, 0.0]), np.array([0.9, 0.1]), 2)
    good = load_balance_loss(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 2)
    assert bad > good


def test_flop_math_and_beam_ratio():
    rep = flop_report(32, 64, 8, 2, 256)
    assert rep["active_fraction"] == 2 / 8
    assert abs(rep["speedup"] - 4.0) < 1e-9
    assert rep["sparse_flops"] * 4 == rep["dense_flops"]
    # Beam's published numbers: 23B active of 501B total
    assert abs(23.0 / 501.0 - 0.0459) < 1e-3


def test_capacity_dropping():
    g = TopKGate(d_model=4, n_experts=2, k=1, seed=0)
    x = np.random.default_rng(0).normal(size=(20, 4))
    gate, idx, _ = g.forward(x)
    keep, dropped = route_with_capacity(gate, idx, capacity_factor=0.5)
    assert 0.0 <= dropped <= 1.0
    # tight capacity must drop something; generous capacity drops nothing
    keep2, dropped2 = route_with_capacity(gate, idx, capacity_factor=10.0)
    assert dropped2 == 0.0
    assert (keep2 == 1.0).all()


def test_reproducible_with_seed():
    def run():
        moe = MoELayer(8, 16, n_experts=4, k=2, seed=11)
        x = np.random.default_rng(0).normal(size=(12, 8))
        out, _ = moe.forward(x)
        return out
    assert np.allclose(run(), run())


def test_param_accounting():
    moe = MoELayer(8, 16, n_experts=4, k=2, seed=0)
    per_expert = 8 * 16 + 16 + 16 * 8 + 8
    assert moe.experts[0].n_params() == per_expert
    assert moe.total_params() == 4 * per_expert + 8 * 4
    assert moe.active_params_per_token() == 2 * per_expert + 8 * 4


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
