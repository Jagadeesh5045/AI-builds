"""Tests for the sparse MoE router demo. Run: python -m pytest -q"""

import math

import pytest

from moe_sparse_router import (
    QWEN4_MAX_ACTIVE_PARAMS_B,
    QWEN4_MAX_TOTAL_PARAMS_B,
    Expert,
    Router,
    SparseMoELayer,
    active_fraction,
    cost_per_task,
    flops_proxy_per_token,
    load_balancing_loss,
    make_tokens,
    softmax,
)


def test_softmax_sums_to_one_and_is_positive():
    probs = softmax([1.0, 2.0, 3.0, 0.5])
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(p > 0 for p in probs)
    assert probs[2] == max(probs)


def test_softmax_rejects_empty():
    with pytest.raises(ValueError):
        softmax([])


def test_expert_output_shape_and_param_count():
    expert = Expert(name="e0", dim=8, hidden=16, seed=1)
    out = expert([0.1] * 8)
    assert len(out) == 8
    assert expert.param_count == 2 * 8 * 16


def test_expert_rejects_wrong_dim():
    expert = Expert(name="e0", dim=8, hidden=16, seed=1)
    with pytest.raises(ValueError):
        expert([0.1] * 4)


def test_router_picks_distinct_topk_with_normalised_weights():
    router = Router(dim=16, num_experts=8, seed=7)
    chosen, weights, probs = router.route([0.2] * 16, top_k=2)
    assert len(chosen) == 2 and len(set(chosen)) == 2
    assert abs(sum(weights) - 1.0) < 1e-9
    assert abs(sum(probs) - 1.0) < 1e-9
    # chosen experts hold the two largest probabilities
    top2 = sorted(range(8), key=lambda i: -probs[i])[:2]
    assert chosen == top2


def test_router_rejects_bad_topk():
    router = Router(dim=8, num_experts=4)
    with pytest.raises(ValueError):
        router.route([0.0] * 8, top_k=0)
    with pytest.raises(ValueError):
        router.route([0.0] * 8, top_k=5)


def test_balanced_loss_is_one():
    # uniform probs and perfectly even assignments give exactly 1.0
    probs = [[0.25] * 4 for _ in range(8)]
    assignments = [[0], [1], [2], [3], [0], [1], [2], [3]]
    assert load_balancing_loss(probs, assignments, 4) == pytest.approx(1.0)


def test_collapsed_router_loss_is_high():
    # every token routed to expert 0 with full confidence
    probs = [[1.0, 0.0, 0.0, 0.0] for _ in range(8)]
    assignments = [[0]] * 8
    assert load_balancing_loss(probs, assignments, 4) == pytest.approx(4.0)


def test_layer_runs_and_preserves_shape():
    layer = SparseMoELayer(dim=16, hidden=32, num_experts=8, top_k=2)
    tokens = make_tokens(64, 16)
    result = layer(tokens)
    assert len(result.outputs) == 64
    assert all(len(o) == 16 for o in result.outputs)
    assert sum(result.expert_load) + result.dropped == 64 * 2


def test_layer_active_params_below_total():
    layer = SparseMoELayer()
    assert layer.active_params < layer.total_params
    assert 0 < layer.active_params / layer.total_params < 0.5


def test_capacity_is_enforced():
    # capacity factor 1.0 with top-1 routing: no expert exceeds the cap
    layer = SparseMoELayer(dim=8, hidden=16, num_experts=4, top_k=1,
                           capacity_factor=1.0)
    tokens = make_tokens(40, 8)
    result = layer(tokens)
    assert max(result.expert_load) <= layer.capacity(40)


def test_layer_rejects_empty_and_bad_capacity():
    with pytest.raises(ValueError):
        SparseMoELayer(capacity_factor=0.5)
    layer = SparseMoELayer()
    with pytest.raises(ValueError):
        layer([])


def test_active_fraction_matches_reported_qwen4_max():
    frac = active_fraction(QWEN4_MAX_TOTAL_PARAMS_B, QWEN4_MAX_ACTIVE_PARAMS_B)
    assert frac == pytest.approx(71.0 / 2400.0)
    assert frac < 0.03


def test_active_fraction_validation():
    with pytest.raises(ValueError):
        active_fraction(0, 1)
    with pytest.raises(ValueError):
        active_fraction(100, 101)
    with pytest.raises(ValueError):
        active_fraction(100, 0)


def test_flops_proxy_scales_with_active_params():
    assert flops_proxy_per_token(71.0) == pytest.approx(142e9)
    assert flops_proxy_per_token(7.1) == pytest.approx(14.2e9)


def test_cost_per_task_scales_with_active_size():
    # same tokens and price: a model activating a tenth of the
    # reference's parameters costs a tenth as much in this proxy
    full = cost_per_task(1_000_000, 1.50, 71.0, 71.0)
    tenth = cost_per_task(1_000_000, 1.50, 7.1, 71.0)
    assert full == pytest.approx(1.50)
    assert tenth == pytest.approx(0.15)
    assert math.isfinite(full) and math.isfinite(tenth)
