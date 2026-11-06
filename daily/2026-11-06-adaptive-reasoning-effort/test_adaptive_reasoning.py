"""Tests for adaptive_reasoning.py - run: python3 -m pytest -q"""

import random

from adaptive_reasoning import (
    Problem, apply_op, majority_vote, make_problem, make_problem_set,
    reasoning_sample, run_benchmark, solve_adaptive, solve_fixed,
    tokens_for,
)


def test_apply_op():
    assert apply_op(5, "+", 4) == 9
    assert apply_op(5, "-", 4) == 1
    assert apply_op(5, "*", 4) == 20


def test_problem_answer_matches_chain():
    rng = random.Random(7)
    for _ in range(50):
        p = make_problem(rng, steps=4)
        value = p.start
        for op, operand in p.ops:
            value = apply_op(value, op, operand)
        assert value == p.answer
        assert p.steps == 4


def test_clean_sample_when_no_slip_possible():
    # Patch the slip probability to zero via a rigged RNG: random()
    # always returns 0.99, which is never < STEP_ERROR_P.
    class NoSlip(random.Random):
        def random(self):
            return 0.99

    p = make_problem_set(1, seed=3)[0]
    assert reasoning_sample(p, NoSlip()) == p.answer


def test_majority_vote_confidence():
    vote = majority_vote([10, 10, 4, 10, 4])
    assert vote.answer == 10
    assert vote.confidence == 0.6
    assert vote.samples == 5


def test_adaptive_respects_ladder_bounds():
    problems = make_problem_set(60, seed=11)
    for i, p in enumerate(problems):
        res = solve_adaptive(p, random.Random(100 + i))
        assert 3 <= res.samples_used <= 9
        assert res.tokens_used == tokens_for(p, res.samples_used)


def test_adaptive_unanimous_stops_at_first_stage():
    class NoSlip(random.Random):
        def random(self):
            return 0.99

    p = make_problem_set(1, seed=5)[0]
    res = solve_adaptive(p, NoSlip())
    assert res.samples_used == 3          # unanimous 3/3, no escalation
    assert res.confidence == 1.0
    assert res.answer == p.answer


def test_fixed_strategies_sample_counts():
    p = make_problem_set(1, seed=9)[0]
    assert solve_fixed(p, 1, random.Random(1)).samples_used == 1
    assert solve_fixed(p, 9, random.Random(1)).samples_used == 9


def test_benchmark_deterministic():
    a = run_benchmark(n=40, seed=42)
    b = run_benchmark(n=40, seed=42)
    for name in a:
        assert a[name].accuracy == b[name].accuracy
        assert a[name].samples == b[name].samples


def test_adaptive_beats_fixed_low_and_saves_vs_fixed_high():
    results = run_benchmark()
    low, high, ad = (results["fixed_low"], results["fixed_high"],
                     results["adaptive"])
    # Voting clearly beats a single noisy sample.
    assert ad.accuracy > low.accuracy + 0.05
    # Adaptive stays within a few points of the always-9 strategy...
    assert ad.accuracy >= high.accuracy - 0.04
    # ...while spending materially fewer samples on average.
    assert ad.avg_samples < high.avg_samples - 2.0


def test_problem_class_shape():
    p = Problem(start=3, ops=[("+", 2), ("*", 4)], answer=20)
    assert p.steps == 2
