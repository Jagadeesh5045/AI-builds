"""ThinkingBox-style state grading: why "looked complete" is not "did complete".

Trending context (8 Nov 2026): Microsoft published an expanded ThinkingBox
analysis on 6 Nov 2026 on agents that edit business records. Across 121,680
valid trials, 79,853 runs failed the executable (final-state) checks, and
67.24% of those failures still showed all three common transcript signals of
success: a completion-like final message, at least one state-changing tool
call, and no explicit final tool error. Their 507-task benchmark also repeats
every task 20 times and scores tasks passed on *every* attempt: Claude Opus
5.5 managed that on 241 tasks, an unmodified Qwen3.8-27B on only 38, and 50
steps of RL with terminal-state rewards lifted the Qwen checkpoint to 107
while cutting the estimated cost per dependable task from $24.36 to $8.00.
(All figures are Microsoft's, as reported by SXF Brief, 8 Nov 2026.)

This demo rebuilds the measurement, in miniature and with the standard
library only:

1. A tiny business-records backend (orders / bookings / permissions).
2. Tasks defined as executable checks over the FINAL state: goal fields must
   hold exact values, every other record must be untouched, and a refund
   policy cap must hold.
3. A simulated agent with labelled failure modes (wrong record, partial
   update, side effect, policy breach, no-op claim, honest partial, visible
   error). Every run produces a transcript AND a final state.
4. Two graders: a transcript grader using exactly Microsoft's three signals,
   and the state grader running the executable checks. The gap between them
   is the false-complete rate.
5. A repeat-reliability experiment in the ThinkingBox style: 3 agent
   profiles x 24 tasks x 20 attempts, scored by pass@1, tasks passed on all
   20 attempts ("dependable tasks"), and cost per dependable task. The
   profile with the best pass@1 is not the most dependable one: what matters
   is where the failures live. An agent that fails 5% of attempts on EVERY
   task passes almost no task 20 times out of 20, while an agent whose
   failures are concentrated on a few hopeless tasks - and is near-perfect
   on the rest - banks dependable tasks even at a lower pass@1.

Run:  python3 state_grading.py
Test: python3 -m pytest test_state_grading.py -q   (or plain unittest)
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Backend: the smallest business-records store that can be wrong in
# interesting ways. Records are plain dicts; a "state" is {id: record}.
# --------------------------------------------------------------------------


def fresh_state() -> dict:
    """A deterministic starting database for one task attempt."""
    return {
        "ORD-101": {"kind": "order", "status": "placed", "refund_pence": 0},
        "ORD-102": {"kind": "order", "status": "placed", "refund_pence": 0},
        "ORD-103": {"kind": "order", "status": "shipped", "refund_pence": 0},
        "BKG-201": {"kind": "booking", "status": "held", "refund_pence": 0},
        "BKG-202": {"kind": "booking", "status": "held", "refund_pence": 0},
        "PRM-301": {"kind": "permission", "status": "viewer", "refund_pence": 0},
        "PRM-302": {"kind": "permission", "status": "viewer", "refund_pence": 0},
    }


@dataclass
class Task:
    """A task is a spec over final state, not a hope about the transcript.

    goals:     (record_id, field, expected_value) that must hold at the end.
    max_refund_pence: policy cap applied to every record's refund field.
    """

    task_id: str
    instruction: str
    goals: list
    max_refund_pence: int = 5000


TASKS = [
    Task("T1", "Cancel ORD-101 and refund 2500p",
         [("ORD-101", "status", "cancelled"), ("ORD-101", "refund_pence", 2500)]),
    Task("T2", "Confirm booking BKG-201",
         [("BKG-201", "status", "confirmed")]),
    Task("T3", "Promote PRM-301 to editor and cancel ORD-103 with a 1000p refund",
         [("PRM-301", "status", "editor"), ("ORD-103", "status", "cancelled"),
          ("ORD-103", "refund_pence", 1000)]),
    Task("T4", "Cancel ORD-102 (no refund) and confirm BKG-202",
         [("ORD-102", "status", "cancelled"), ("BKG-202", "status", "confirmed")]),
]


def grade_state(initial: dict, final: dict, task: Task) -> tuple:
    """Executable checks over final state. Returns (passed, failures)."""
    failures = []
    goal_records = {g[0] for g in task.goals}
    for rec_id, fld, expected in task.goals:
        actual = final.get(rec_id, {}).get(fld)
        if actual != expected:
            failures.append(f"{rec_id}.{fld}: expected {expected!r}, got {actual!r}")
    # No side effects: any record not named in the goals must be identical.
    for rec_id, before in initial.items():
        if rec_id in goal_records:
            continue
        if final.get(rec_id) != before:
            failures.append(f"side effect on untouched record {rec_id}")
    # Policy: the refund cap holds on every record, goal records included.
    for rec_id, rec in final.items():
        if rec.get("refund_pence", 0) > task.max_refund_pence:
            failures.append(
                f"policy breach: {rec_id} refund {rec['refund_pence']}p "
                f"> cap {task.max_refund_pence}p")
    return (not failures), failures


# --------------------------------------------------------------------------
# Simulated agent: one run = behaviour -> (transcript, final state).
# The agent is deliberately ordinary: most of its failures are polite,
# confident and error-free on the surface - that is the whole problem.
# --------------------------------------------------------------------------

COMPLETION_PHRASES = ("done", "complete", "updated", "all set", "confirmed and")


def completion_like(message: str) -> bool:
    m = (message or "").lower()
    return any(p in m for p in COMPLETION_PHRASES)


@dataclass
class RunResult:
    behaviour: str
    final_message: str
    state_changing_calls: int
    final_tool_error: bool
    final_state: dict


# Behaviour mix for the "typical agent" in experiment 1 (weights sum to 1).
BEHAVIOUR_MIX = [
    ("correct", 0.40),
    ("wrong_record", 0.13),
    ("partial_claim", 0.10),
    ("side_effect", 0.09),
    ("policy_breach", 0.07),
    ("noop_claim", 0.05),
    ("honest_partial", 0.06),
    ("visible_error", 0.05),
    ("crash_no_message", 0.05),
]

NEIGHBOUR = {"ORD-101": "ORD-102", "ORD-102": "ORD-101", "ORD-103": "ORD-102",
             "BKG-201": "BKG-202", "BKG-202": "BKG-201",
             "PRM-301": "PRM-302", "PRM-302": "PRM-301"}


def _apply(state, rec_id, fld, value):
    state[rec_id][fld] = value


def run_agent(task: Task, behaviour: str, initial: dict) -> RunResult:
    state = copy.deepcopy(initial)
    calls = 0
    if behaviour == "correct":
        for rec_id, fld, val in task.goals:
            _apply(state, rec_id, fld, val)
            calls += 1
        return RunResult(behaviour, "Done - all updates complete.", calls, False, state)
    if behaviour == "wrong_record":
        for rec_id, fld, val in task.goals:
            _apply(state, NEIGHBOUR[rec_id], fld, val)
            calls += 1
        return RunResult(behaviour, "Done - record updated as requested.", calls, False, state)
    if behaviour == "partial_claim":
        rec_id, fld, val = task.goals[0]
        _apply(state, rec_id, fld, val)
        return RunResult(behaviour, "Done - everything is updated and complete.",
                         1, False, state)
    if behaviour == "side_effect":
        for rec_id, fld, val in task.goals:
            _apply(state, rec_id, fld, val)
            calls += 1
        other = next(r for r in state if r not in {g[0] for g in task.goals})
        _apply(state, other, "status", "cancelled")
        return RunResult(behaviour, "Done - all set, updates complete.", calls + 1, False, state)
    if behaviour == "policy_breach":
        for rec_id, fld, val in task.goals:
            _apply(state, rec_id, fld, val)
            calls += 1
        rec_id = task.goals[0][0]
        state[rec_id]["refund_pence"] = task.max_refund_pence + 2500
        return RunResult(behaviour, "Done - refund issued, all complete.", calls, False, state)
    if behaviour == "noop_claim":
        rec_id = task.goals[0][0]
        state[rec_id]["touched"] = True  # a state-changing call that changes nothing that matters
        return RunResult(behaviour, "Done - I have updated the record.", 1, False, state)
    if behaviour == "honest_partial":
        rec_id, fld, val = task.goals[0]
        _apply(state, rec_id, fld, val)
        return RunResult(behaviour,
                         "I managed the first update but could not finish the rest.",
                         1, False, state)
    if behaviour == "visible_error":
        return RunResult(behaviour,
                         "The update tool returned an error, so nothing was saved.",
                         0, True, state)
    if behaviour == "crash_no_message":
        return RunResult(behaviour, "", 0, False, state)
    raise ValueError(f"unknown behaviour {behaviour!r}")


def pick_behaviour(rng: random.Random) -> str:
    x, acc = rng.random(), 0.0
    for name, w in BEHAVIOUR_MIX:
        acc += w
        if x < acc:
            return name
    return BEHAVIOUR_MIX[-1][0]


# --------------------------------------------------------------------------
# The two graders. The transcript grader uses exactly the three signals
# Microsoft found misleading: completion-like final message, at least one
# state-changing tool call, no explicit final tool error.
# --------------------------------------------------------------------------


def grade_transcript(run: RunResult) -> bool:
    return (completion_like(run.final_message)
            and run.state_changing_calls >= 1
            and not run.final_tool_error)


def experiment_false_complete(runs: int = 600, seed: int = 8) -> dict:
    rng = random.Random(seed)
    state_pass = state_fail = false_complete = transcript_pass = 0
    for i in range(runs):
        task = TASKS[i % len(TASKS)]
        initial = fresh_state()
        run = run_agent(task, pick_behaviour(rng), initial)
        ok, _ = grade_state(initial, run.final_state, task)
        t_ok = grade_transcript(run)
        transcript_pass += t_ok
        if ok:
            state_pass += 1
        else:
            state_fail += 1
            if t_ok:
                false_complete += 1
    return {
        "runs": runs,
        "state_pass_rate": state_pass / runs,
        "transcript_pass_rate": transcript_pass / runs,
        "failed_runs": state_fail,
        "false_complete": false_complete,
        "false_complete_rate": (false_complete / state_fail) if state_fail else 0.0,
    }


# --------------------------------------------------------------------------
# Experiment 2: repeat reliability, ThinkingBox style. Each profile has a
# per-task success probability; a task is "dependable" only if all 20
# attempts pass. Cost per dependable task = total spend / dependable tasks,
# which punishes spiky agents exactly the way Microsoft's $24.36 -> $8.00
# figure rewards fixing them.
# --------------------------------------------------------------------------

ATTEMPT_COST = 0.60  # dollars per attempt, same for every profile
REPEATS = 20


def _profile_task_probs(profile: str, n_tasks: int, rng: random.Random) -> list:
    if profile == "even":
        # Best pass@1 of the three - but its failures are spread thinly over
        # every task, so almost no task survives 20 repeats unscathed.
        return [0.95 + rng.uniform(-0.01, 0.01) for _ in range(n_tasks)]
    if profile == "concentrated":
        # Slightly worse pass@1, but three quarters of its tasks are
        # near-perfect; the failures pile up on a hopeless quarter.
        return [0.999 if rng.random() < 0.75 else 0.70 for _ in range(n_tasks)]
    if profile == "steady":
        # Evenly mediocre: lower pass@1 and the same thin-spread problem.
        return [0.88 + rng.uniform(-0.02, 0.02) for _ in range(n_tasks)]
    raise ValueError(profile)


def experiment_reliability(n_tasks: int = 24, seed: int = 11) -> dict:
    out = {}
    for profile in ("even", "concentrated", "steady"):
        rng = random.Random(hash((seed, profile)) & 0xFFFFFFFF)
        probs = _profile_task_probs(profile, n_tasks, rng)
        attempts = passes = dependable = 0
        for p in probs:
            task_passes = sum(1 for _ in range(REPEATS) if rng.random() < p)
            attempts += REPEATS
            passes += task_passes
            dependable += (task_passes == REPEATS)
        spend = attempts * ATTEMPT_COST
        out[profile] = {
            "pass_at_1": passes / attempts,
            "dependable_tasks": dependable,
            "n_tasks": n_tasks,
            "cost_per_attempt_success": spend / passes if passes else float("inf"),
            "cost_per_dependable_task": (spend / dependable) if dependable else float("inf"),
        }
    return out


def main() -> None:
    print("=== Experiment 1: transcript grading vs state grading ===")
    r = experiment_false_complete()
    print(f"runs:                 {r['runs']}")
    print(f"state grader passes:  {r['state_pass_rate']:.1%}")
    print(f"transcript passes:    {r['transcript_pass_rate']:.1%}")
    print(f"failed runs:          {r['failed_runs']}")
    print(f"of the failures, {r['false_complete']} still looked complete "
          f"({r['false_complete_rate']:.1%})")
    print("Microsoft's ThinkingBox figure for the same quantity: 67.24% "
          "of 79,853 failed runs (121,680 trials, 6 Nov 2026).")
    print()
    print("=== Experiment 2: pass@1 vs dependable tasks (20 repeats) ===")
    rel = experiment_reliability()
    for name, s in rel.items():
        print(f"{name:8s} pass@1 {s['pass_at_1']:.3f} | dependable "
              f"{s['dependable_tasks']}/{s['n_tasks']} tasks | "
              f"${s['cost_per_attempt_success']:.2f} per successful attempt | "
              f"${s['cost_per_dependable_task']:.2f} per dependable task")
    best1 = max(rel, key=lambda k: rel[k]["pass_at_1"])
    bestd = max(rel, key=lambda k: rel[k]["dependable_tasks"])
    print(f"best pass@1: {best1}; most dependable: {bestd} "
          f"({'same agent' if best1 == bestd else 'DIFFERENT agent - the ranking flips'})")


if __name__ == "__main__":
    main()
