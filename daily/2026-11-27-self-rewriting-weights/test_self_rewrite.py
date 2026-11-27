"""Tests for the self-rewriting weights miniature."""

import numpy as np

from self_rewrite import (
    accuracy,
    diagnose_difference,
    gate_and_submit,
    make_direction,
    make_misaligned_head,
    make_traces,
    propose_edit,
    random_edit,
    run_demo,
)


def test_diagnosis_recovers_true_direction():
    rng = np.random.default_rng(3)
    direction = make_direction(rng)
    X, y = make_traces(rng, 800, direction)
    diff = diagnose_difference(X, y)
    cos = float(diff @ direction / np.linalg.norm(diff))
    assert cos > 0.85, cos


def test_misaligned_head_starts_weak():
    rng = np.random.default_rng(5)
    direction = make_direction(rng)
    w, b = make_misaligned_head(rng, direction)
    X, y = make_traces(rng, 600, direction)
    assert accuracy(w, b, X, y) < 0.75


def test_self_edit_improves_held_out_accuracy():
    out = run_demo(seed=7, verbose=False)
    assert out["submitted"] is True
    assert out["self_test"] > out["base_test"] + 0.15
    assert out["self_test"] > 0.85


def test_gate_rejects_random_noise_edit():
    out = run_demo(seed=7, verbose=False)
    assert out["rand_submitted"] is False


def test_gate_reverts_a_harmful_edit():
    rng = np.random.default_rng(11)
    direction = make_direction(rng)
    w, b = make_misaligned_head(rng, direction)
    X, y = make_traces(rng, 400, direction)
    diff = diagnose_difference(X, y)
    harmful = propose_edit(w, diff, alpha=-3.0)  # edit the wrong way
    kept, submitted, _, _ = gate_and_submit(w, b, harmful, X, y)
    assert submitted is False
    assert np.array_equal(kept, w)


def test_random_edit_does_not_fix_the_head():
    rng = np.random.default_rng(13)
    direction = make_direction(rng)
    w, b = make_misaligned_head(rng, direction)
    X, y = make_traces(rng, 600, direction)
    base = accuracy(w, b, X, y)
    w_rand = random_edit(w, rng, norm=2.0)
    assert accuracy(w_rand, b, X, y) < base + 0.15


def test_edit_is_rank1_direction_only():
    rng = np.random.default_rng(17)
    direction = make_direction(rng)
    w, _ = make_misaligned_head(rng, direction)
    X, y = make_traces(rng, 200, direction)
    diff = diagnose_difference(X, y)
    edited = propose_edit(w, diff, alpha=1.0)
    delta = edited - w
    cos = float(delta @ diff / (np.linalg.norm(delta) * np.linalg.norm(diff)))
    assert abs(cos - 1.0) < 1e-9  # the whole edit lies along the diagnosis
