"""Tests for the looped-transformer demo. Run: python3 -m pytest
test_looped_transformer.py  (or plain python3, see __main__)."""

import math

from looped_transformer import (
    LoopResult,
    StackShape,
    inkling2_mini_shape,
    reachable_within,
    run_looped,
    shared_block,
)


def _source(n=15, at=0):
    s = [0.0] * n
    s[at] = 1.0
    return s


def test_one_loop_reaches_only_the_source():
    r = run_looped(_source(), 1)
    # the first application is the injection itself: only the source
    # cell carries signal; propagation starts on the second loop
    assert reachable_within(r.state) == 1
    assert r.state[1] == 0.0


def test_signal_travels_exactly_one_hop_per_loop():
    for loops in (2, 3, 5):
        r = run_looped(_source(), loops)
        assert reachable_within(r.state) == loops
        assert r.state[loops] == 0.0


def test_far_end_needs_diameter_loops():
    short = run_looped(_source(), 7)
    full = run_looped(_source(), 15)
    assert short.state[-1] == 0.0
    assert full.state[-1] > 0.0


def test_signal_does_not_decay_with_distance():
    r = run_looped(_source(), 15)
    # max-propagation: the far end carries the full source strength;
    # an averaging block would leave it at 0.5**14 ~= 0.00006
    assert all(0.0 <= v <= 1.0 for v in r.state)
    assert r.state[1] == 1.0
    assert r.state[-1] == 1.0


def test_shared_block_is_deterministic_and_pure():
    s = _source()
    a = shared_block([0.0] * 15, s)
    b = shared_block([0.0] * 15, s)
    assert a == b


def test_more_loops_never_lose_reach():
    prev = 0
    for loops in range(0, 15):
        got = reachable_within(run_looped(_source(), loops).state)
        assert got >= prev
        prev = got


def test_adaptive_halting_matches_fixed_far_end():
    fixed = run_looped(_source(), 15)
    adapt = run_looped(_source(), 15, halt_epsilon=0.01)
    assert adapt.state == fixed.state


def test_adaptive_does_not_halt_unreached_positions():
    # with a 3-loop budget the far end is unreached; it must still be
    # marked active throughout (loops_used == budget), not halted at 1
    adapt = run_looped(_source(), 3, halt_epsilon=0.01)
    assert adapt.loops_used[-1] == 3
    assert adapt.state[-1] == 0.0


def test_adaptive_halting_saves_applications():
    fixed = run_looped(_source(), 15)
    adapt = run_looped(_source(), 15, halt_epsilon=0.01)
    assert adapt.total_applications < fixed.total_applications
    # near-source positions halt strictly earlier than the far end
    assert adapt.loops_used[0] < adapt.loops_used[-1]


def test_adaptive_never_exceeds_budget():
    adapt = run_looped(_source(), 6, halt_epsilon=0.01)
    assert all(u <= 6 for u in adapt.loops_used)
    assert adapt.total_applications <= 6 * 15


def test_inkling_shape_arithmetic():
    s = inkling2_mini_shape()
    assert s.unique_layers == 24
    assert s.applications == 48  # 6 + 12*3 + 6


def test_saving_factor():
    s = inkling2_mini_shape()
    assert s.saving_factor() == 2.0
    plain = StackShape(unique_layers=24, looped_layers=0, loops=3)
    assert plain.applications == 24 and plain.saving_factor() == 1.0


def test_stored_vs_unlooped_params():
    s = inkling2_mini_shape()
    assert s.stored_params(1.0) == 24.0
    assert s.unlooped_params(1.0) == 48.0


def test_centre_source_spreads_both_ways():
    r = run_looped(_source(at=7), 4)
    assert reachable_within(r.state) == 7  # 3 left + source + 3 right


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} tests passing")
