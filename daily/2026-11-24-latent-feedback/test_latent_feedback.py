"""Tests for the Attention Latent Feedback miniature."""
import numpy as np

from latent_feedback import (EMB_DIM, build_dataset, flop_overhead,
                             learn_gates, make_world, model_features, mse,
                             ridge_fit, ridge_predict, DEEP_DIM, SEED)


def _data():
    rng = np.random.default_rng(SEED)
    world = make_world(rng)
    tr = build_dataset(120, rng, world)
    va = build_dataset(30, rng, world)
    te = build_dataset(60, rng, world)
    return tr, va, te


def _split(Xb):
    return Xb[:, :EMB_DIM], Xb[:, EMB_DIM:]


def test_shapes():
    (Xb, d, y), _, _ = _data()
    assert Xb.shape[0] == d.shape[0] == y.shape[0]
    assert d.shape[1] == DEEP_DIM


def test_determinism():
    a = _data()
    b = _data()
    assert np.allclose(a[0][2], b[0][2])


def test_alf_beats_baseline():
    (Xb_tr, d_tr, y_tr), _, (Xb_te, d_te, y_te) = _data()
    cur_tr, prev_tr = _split(Xb_tr)
    cur_te, prev_te = _split(Xb_te)
    w = ridge_fit(model_features(cur_tr, prev_tr), y_tr)
    base = mse(ridge_predict(model_features(cur_te, prev_te), w), y_te)
    w2 = ridge_fit(model_features(cur_tr, d_tr), y_tr)
    alf = mse(ridge_predict(model_features(cur_te, d_te), w2), y_te)
    # The label is built from the previous token's deep state, so access
    # to it must cut error substantially, not marginally.
    assert alf < 0.5 * base


def test_gates_in_range_and_help():
    (Xb_tr, d_tr, y_tr), (Xb_va, d_va, y_va), (Xb_te, d_te, y_te) = _data()
    cur_tr, prev_tr = _split(Xb_tr)
    cur_va, prev_va = _split(Xb_va)
    cur_te, prev_te = _split(Xb_te)
    gates, expand = learn_gates(cur_tr, d_tr, y_tr, cur_va, d_va, y_va,
                                prev_va, steps=150)
    assert gates.shape == (3,)
    assert np.all(gates >= 0.0) and np.all(gates <= 1.5)
    w = ridge_fit(model_features(cur_tr, expand(d_tr, gates)), y_tr)
    gated = mse(ridge_predict(model_features(cur_te, expand(d_te, gates)), w),
                y_te)
    wb = ridge_fit(model_features(cur_tr, prev_tr), y_tr)
    base = mse(ridge_predict(model_features(cur_te, prev_te), wb), y_te)
    assert gated < base


def test_overhead_is_small():
    # The feedback read is a vector blend, not a second attention pass:
    # its share of layer FLOPs must stay in low single digits.
    assert 0.0 < flop_overhead() < 0.05


def test_zero_gate_recovers_baseline_information():
    # With every gate at zero the feedback carries nothing, so the model
    # must do no better than (roughly) the baseline.
    (Xb_tr, d_tr, y_tr), _, (Xb_te, d_te, y_te) = _data()
    cur_tr, prev_tr = _split(Xb_tr)
    cur_te, prev_te = _split(Xb_te)
    zero_tr = np.zeros_like(d_tr)
    zero_te = np.zeros_like(d_te)
    w = ridge_fit(model_features(cur_tr, zero_tr), y_tr)
    gated_off = mse(ridge_predict(model_features(cur_te, zero_te), w), y_te)
    wb = ridge_fit(model_features(cur_tr, prev_tr), y_tr)
    base = mse(ridge_predict(model_features(cur_te, prev_te), wb), y_te)
    assert gated_off >= 0.9 * base
