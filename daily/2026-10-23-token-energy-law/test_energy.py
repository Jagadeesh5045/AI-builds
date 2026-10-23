"""Tests for the token-energy accounting toy.

These pin down the *structural* claims of the model (scaling laws,
ordering relations, speculative breakeven), which hold for any choice
of the illustrative hardware constants. Absolute joule figures are
deliberately not asserted.

Runs with plain `python3 test_energy.py` (no pytest needed).
"""
import math
from contextlib import contextmanager

import numpy as np

from energy import (
    ModelConfig,
    active_bytes,
    flops_per_token,
    joules_per_token,
    kv_bytes,
    lm_head_bytes,
    realized_time,
    speculative_breakeven_alpha,
    speculative_per_token,
    weights_bytes,
)
import energy as E

CFG_7B = ModelConfig("7B-fp16", params=7_000_000_000, layers=32, d_model=4096, dtype_bytes=2.0)
CFG_7B_Q4 = ModelConfig("7B-q4", params=7_000_000_000, layers=32, d_model=4096, dtype_bytes=0.5)
CFG_70B = ModelConfig("70B-fp16", params=70_000_000_000, layers=80, d_model=8192, dtype_bytes=2.0)
CFG_1B = ModelConfig("1B-fp16", params=1_000_000_000, layers=24, d_model=2048, dtype_bytes=2.0)


def approx(a, b, rel=1e-9):
    assert abs(a - b) <= rel * max(1.0, abs(a), abs(b)), f"{a} !~= {b}"


@contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_weights_scale_with_params():
    approx(weights_bytes(CFG_70B), 10 * weights_bytes(CFG_7B))


def test_quantized_moves_fewer_bytes():
    approx(weights_bytes(CFG_7B_Q4), weights_bytes(CFG_7B) / 4)


def test_kv_bytes_zero_at_empty_context():
    assert kv_bytes(CFG_7B, 0) == 0.0


def test_kv_bytes_linear_in_context():
    approx(kv_bytes(CFG_7B, 2048), 2 * kv_bytes(CFG_7B, 1024))
    # The whole active-bytes term is linear in ctx, so first differences match.
    d1 = active_bytes(CFG_7B, 2048) - active_bytes(CFG_7B, 1024)
    d2 = active_bytes(CFG_7B, 1024) - active_bytes(CFG_7B, 0)
    approx(d1, d2)


def test_active_bytes_splits_additively():
    cfg, ctx = CFG_7B, 4096
    approx(active_bytes(cfg, ctx), weights_bytes(cfg) + kv_bytes(cfg, ctx) + lm_head_bytes(cfg))


def test_larger_model_moves_more_bytes_and_joules():
    for ctx in (0, 1024, 32768):
        assert active_bytes(CFG_70B, ctx) > active_bytes(CFG_7B, ctx)
        assert joules_per_token(CFG_70B, ctx) > joules_per_token(CFG_7B, ctx)


def test_quantized_costs_less_than_fp16():
    for ctx in (0, 8192):
        assert joules_per_token(CFG_7B_Q4, ctx) < joules_per_token(CFG_7B, ctx)


def test_joules_increase_with_context():
    assert joules_per_token(CFG_7B, 32768) > joules_per_token(CFG_7B, 1024) > joules_per_token(
        CFG_7B, 0
    )


def test_decode_is_memory_bound():
    # Realized time is dominated by bytes/bandwidth, not the compute tail.
    mem_time = active_bytes(CFG_7B, 8192) / E.BANDWIDTH
    total = realized_time(CFG_7B, 8192)
    assert mem_time <= total < 2 * mem_time


def test_joules_fit_by_bytes_and_flops():
    # The toy law by construction: E = a*bytes + b*flops. A two-term fit
    # must recover it exactly.
    cfgs = [CFG_1B, CFG_7B, CFG_7B_Q4, CFG_70B]
    xs = np.array([[active_bytes(c, 4096), flops_per_token(c, 4096)] for c in cfgs])
    ys = np.array([joules_per_token(c, 4096) for c in cfgs])
    coef, *_ = np.linalg.lstsq(np.c_[xs, np.ones(4)], ys, rcond=None)
    r2 = 1 - np.sum((ys - np.c_[xs, np.ones(4)] @ coef) ** 2) / np.sum((ys - ys.mean()) ** 2)
    assert r2 > 0.999999, r2
    approx(coef[0], E.E_PER_BYTE, rel=1e-6)
    approx(coef[1], E.E_PER_FLOP, rel=1e-6)


def test_bytes_alone_underfits_quantized_configs():
    # Honest structural check: bytes alone cannot explain quantized
    # configs (same flops, 4x fewer bytes). The flops term is needed.
    cfgs = [CFG_1B, CFG_7B, CFG_7B_Q4, CFG_70B]
    xs = np.array([active_bytes(c, 4096) for c in cfgs])
    ys = np.array([joules_per_token(c, 4096) for c in cfgs])
    a, b = np.polyfit(xs, ys, 1)
    r2 = 1 - np.sum((ys - (a * xs + b)) ** 2) / np.sum((ys - ys.mean()) ** 2)
    assert r2 < 0.999, r2  # bytes alone is NOT the whole story


def test_flops_attention_term_grows_with_context():
    assert flops_per_token(CFG_7B, 8192) > flops_per_token(CFG_7B, 1024)
    # ...but weights dominate: 8x ctx barely moves total flops.
    assert flops_per_token(CFG_7B, 8192) < 1.5 * flops_per_token(CFG_7B, 1024)


def test_speculative_wins_at_full_acceptance():
    base = joules_per_token(CFG_7B, 4096)
    assert speculative_per_token(CFG_7B, CFG_1B, 4096, k=4, alpha=1.0) < base


def test_speculative_loses_at_low_acceptance():
    base = joules_per_token(CFG_7B, 4096)
    assert speculative_per_token(CFG_7B, CFG_1B, 4096, k=4, alpha=0.1) > base


def test_speculative_monotone_in_alpha():
    lo = speculative_per_token(CFG_7B, CFG_1B, 4096, k=4, alpha=0.5)
    hi = speculative_per_token(CFG_7B, CFG_1B, 4096, k=4, alpha=0.9)
    assert lo > hi


def test_breakeven_alpha_in_unit_interval_for_cheap_draft():
    a_star = speculative_breakeven_alpha(CFG_7B, CFG_1B, 4096, k=4)
    assert 0.0 < a_star < 1.0
    # At alpha* speculation ties the baseline, by definition.
    approx(
        speculative_per_token(CFG_7B, CFG_1B, 4096, k=4, alpha=a_star),
        joules_per_token(CFG_7B, 4096),
    )


def test_breakeven_alpha_above_one_when_draft_is_useless():
    # Draft as expensive as the target: speculation can never win.
    assert speculative_breakeven_alpha(CFG_7B, CFG_7B, 4096, k=4) > 1.0


def test_invalid_inputs_rejected():
    with raises(ValueError):
        speculative_per_token(CFG_7B, CFG_1B, 4096, k=4, alpha=0.0)
    with raises(ValueError):
        speculative_per_token(CFG_7B, CFG_1B, 4096, k=4, alpha=1.5)
    with raises(ValueError):
        speculative_per_token(CFG_7B, CFG_1B, 4096, k=0, alpha=1.0)
    with raises(ValueError):
        kv_bytes(CFG_7B, -1)


def test_all_positive():
    for cfg in (CFG_1B, CFG_7B, CFG_7B_Q4, CFG_70B):
        for ctx in (0, 512):
            assert joules_per_token(cfg, ctx) > 0
            assert realized_time(cfg, ctx) > 0
            assert math.isfinite(joules_per_token(cfg, ctx))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passing")
    raise SystemExit(1 if failed else 0)
