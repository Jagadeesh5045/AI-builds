"""Tests for the ThinkingBox-style state-grading demo.

Run: python3 -m pytest test_state_grading.py -q
"""
import copy

from state_grading import (TASKS, experiment_false_complete,
                           experiment_reliability, fresh_state, grade_state,
                           grade_transcript, run_agent)

T1 = TASKS[0]


def _run(behaviour, task=T1):
    initial = fresh_state()
    return initial, run_agent(task, behaviour, initial)


def test_correct_run_passes_both_graders():
    initial, run = _run("correct")
    ok, failures = grade_state(initial, run.final_state, T1)
    assert ok and failures == []
    assert grade_transcript(run) is True


def test_wrong_record_fails_state_but_looks_complete():
    initial, run = _run("wrong_record")
    ok, failures = grade_state(initial, run.final_state, T1)
    assert not ok and failures
    assert grade_transcript(run) is True  # the false-complete problem, live


def test_partial_claim_is_caught_by_state_grader():
    initial, run = _run("partial_claim")
    ok, _ = grade_state(initial, run.final_state, T1)
    assert not ok
    assert grade_transcript(run) is True


def test_side_effect_on_untouched_record_is_caught():
    initial, run = _run("side_effect")
    ok, failures = grade_state(initial, run.final_state, T1)
    assert not ok
    assert any("side effect" in f for f in failures)


def test_policy_breach_is_caught_even_when_goals_met():
    initial, run = _run("policy_breach")
    ok, failures = grade_state(initial, run.final_state, T1)
    assert not ok
    assert any("policy breach" in f for f in failures)


def test_noop_claim_changes_state_but_not_the_goal():
    initial, run = _run("noop_claim")
    ok, _ = grade_state(initial, run.final_state, T1)
    assert not ok
    assert run.state_changing_calls >= 1
    assert grade_transcript(run) is True


def test_visible_error_fails_both_graders():
    initial, run = _run("visible_error")
    ok, _ = grade_state(initial, run.final_state, T1)
    assert not ok
    assert grade_transcript(run) is False


def test_honest_partial_fails_transcript_grader_too():
    initial, run = _run("honest_partial")
    ok, _ = grade_state(initial, run.final_state, T1)
    assert not ok
    assert grade_transcript(run) is False


def test_state_grader_does_not_mutate_inputs():
    initial, run = _run("correct")
    snapshot = copy.deepcopy(run.final_state)
    grade_state(initial, run.final_state, T1)
    assert run.final_state == snapshot


def test_false_complete_rate_is_high_but_not_total():
    r = experiment_false_complete(runs=600, seed=8)
    assert r["failed_runs"] > 200
    assert 0.5 < r["false_complete_rate"] < 1.0
    # the transcript grader must be strictly more lenient than the state one
    assert r["transcript_pass_rate"] > r["state_pass_rate"]


def test_experiment_is_deterministic():
    assert experiment_false_complete(200, 3) == experiment_false_complete(200, 3)
    assert experiment_reliability(12, 5) == experiment_reliability(12, 5)


def test_reliability_ranking_flips():
    rel = experiment_reliability(n_tasks=24, seed=11)
    # "even" has the best single-attempt rate...
    assert rel["even"]["pass_at_1"] > rel["concentrated"]["pass_at_1"]
    # ...but "concentrated" owns far more all-20 dependable tasks and is
    # cheaper per dependable task - the ThinkingBox ranking flip: where the
    # failures live matters more than how many there are.
    assert (rel["concentrated"]["dependable_tasks"]
            > rel["even"]["dependable_tasks"])
    assert (rel["concentrated"]["cost_per_dependable_task"]
            < rel["even"]["cost_per_dependable_task"])
