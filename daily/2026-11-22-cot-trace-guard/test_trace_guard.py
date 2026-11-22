"""Tests for trace_guard. Plain asserts with a tiny runner (no pytest on
this VM); also collectable by pytest if available."""

import random

from trace_guard import (FEATURES, GuardedServer, LabelOnlyAttacker,
                         TraceAttacker, TraceGuard, TriageModel,
                         VulnerableServer, agreement, make_case, run_demo)


def test_model_threshold():
    m = TriageModel()
    hot = {f: 1.0 for f in FEATURES}
    cold = {f: 0.0 for f in FEATURES}
    assert m.decide(hot) == "escalate"
    assert m.decide(cold) == "handle"


def test_vulnerable_trace_contains_rubric():
    s = VulnerableServer(TriageModel())
    out = s.respond(make_case(random.Random(1)))
    for f in FEATURES:
        assert f in out["trace"]


def test_trace_attacker_recovers_exact_weights():
    m = TriageModel()
    s = VulnerableServer(m)
    clone = TraceAttacker().clone_from_trace(
        s.respond(make_case(random.Random(2)))["trace"])
    assert clone.weights == m.weights
    assert clone.threshold == m.threshold


def test_trace_clone_agrees_perfectly():
    m = TriageModel()
    s = VulnerableServer(m)
    clone = TraceAttacker().clone_from_trace(
        s.respond(make_case(random.Random(3)))["trace"])
    assert agreement(m.decide, clone.decide) == 1.0


def test_guard_blocks_digits_and_rubric_words():
    g = TraceGuard()
    assert not g.scan("urgency=0.9 contributes 0.378")
    assert not g.scan("the rubric says escalate")
    assert g.scan("Decision made under the standard triage policy.")


def test_guard_enforce_counts_blocks():
    g = TraceGuard()
    out = g.enforce("score = 0.42*urgency", "safe text")
    assert out == "safe text" and g.blocked == 1
    out = g.enforce("all good here", "safe text")
    assert out == "all good here" and g.blocked == 1


def test_guarded_server_publishes_no_digits():
    s = GuardedServer(TriageModel())
    rng = random.Random(5)
    for _ in range(25):
        out = s.respond(make_case(rng))
        assert not any(ch.isdigit() for ch in out["trace"])
        for f in FEATURES:
            assert f not in out["trace"]


def test_guarded_server_carries_canary():
    s = GuardedServer(TriageModel())
    out = s.respond(make_case(random.Random(6)))
    assert out["canary"].startswith("canary-")


def test_guarded_labels_match_model():
    m = TriageModel()
    s = GuardedServer(m)
    rng = random.Random(8)
    for _ in range(50):
        case = make_case(rng)
        assert s.respond(case)["label"] == m.decide(case)


def test_label_only_attacker_is_imperfect():
    out = run_demo()
    assert out["trace_clone_agreement"] == 1.0
    # Labels alone, same era of budget: a real approximation, not a clone.
    assert 0.70 < out["label_only_agreement"] < 1.0


def test_probing_detector_flags_grid_sweep():
    s = GuardedServer(TriageModel())
    base = {f: 0.5 for f in FEATURES}
    for i in range(24):
        case = dict(base)
        case["urgency"] = round(0.02 * (i + 1), 3)
        s.respond(case)
    assert s.probing_detected()


def test_probing_detector_quiet_on_organic_traffic():
    s = GuardedServer(TriageModel())
    rng = random.Random(11)
    for _ in range(24):
        s.respond(make_case(rng))
    assert not s.probing_detected()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passing")
