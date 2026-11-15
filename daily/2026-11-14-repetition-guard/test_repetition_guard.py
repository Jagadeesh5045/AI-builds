"""Tests for the repetition guard demo. Run: python3 -m pytest -q"""

from repetition_guard import (
    LOOP_BLOCK,
    count_repeated_steps,
    find_repeated_block,
    repetition_rate,
    run_agent,
    run_suite,
)


# --- detector -----------------------------------------------------------

def test_detector_finds_back_to_back_loop():
    trace = ["a", "b"] + LOOP_BLOCK + LOOP_BLOCK + ["c"]
    hit = find_repeated_block(trace)
    assert hit is not None
    assert trace[hit : hit + 3] == LOOP_BLOCK


def test_detector_ignores_unique_trace():
    trace = ["read_source", "extract_fields", "merge_tables", "run_model"]
    assert find_repeated_block(trace) is None


def test_detector_ignores_distant_repeat():
    # Same block, but far apart: reuse, not a loop.
    trace = LOOP_BLOCK + [f"filler_{i}" for i in range(10)] + LOOP_BLOCK
    assert find_repeated_block(trace) is None


def test_detector_short_trace():
    assert find_repeated_block(["a", "b"]) is None
    assert find_repeated_block([]) is None


def test_count_repeated_steps_zero_on_unique():
    assert count_repeated_steps(["a", "b", "c", "d", "e"]) == 0


def test_count_repeated_steps_positive_on_loop():
    trace = LOOP_BLOCK + LOOP_BLOCK + LOOP_BLOCK
    assert count_repeated_steps(trace) > 0


# --- agents -------------------------------------------------------------

def test_unguarded_agent_completes_but_repeats():
    r = run_agent(["fetch", "join", "score", "report"], seed=3, guarded=False)
    assert r.completed
    assert r.repeated_steps > 0


def test_guarded_agent_completes_without_repeats():
    r = run_agent(["fetch", "join", "score", "report"], seed=3, guarded=True)
    assert r.completed
    assert r.repeated_steps == 0
    assert r.loops_detected >= 1
    assert r.replays == r.loops_detected


def test_guarded_uses_fewer_steps_on_sticky_workload():
    tasks = ["join", "score", "join", "score"]
    b = run_agent(tasks, seed=5, guarded=False)
    g = run_agent(tasks, seed=5, guarded=True)
    assert g.steps < b.steps


def test_non_sticky_workload_identical():
    tasks = ["fetch", "parse", "report", "verify"]
    b = run_agent(tasks, seed=5, guarded=False)
    g = run_agent(tasks, seed=5, guarded=True)
    assert b.trace == g.trace
    assert repetition_rate(b) == 0.0


# --- suite --------------------------------------------------------------

def test_suite_guarded_repetition_near_zero():
    base, guard = run_suite(n_tasks=20, seed=11)
    assert repetition_rate(guard) < 0.02
    assert repetition_rate(base) > 0.10


def test_suite_guard_saves_budget():
    base, guard = run_suite(n_tasks=20, seed=11)
    assert guard.steps < base.steps
    assert guard.loops_detected > 0
