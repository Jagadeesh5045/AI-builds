# ThinkingBox state grading: "looked complete" is not "did complete"

**Trending:** Microsoft published an expanded **ThinkingBox** analysis on
6 Nov 2026 (covered in SXF Brief's AI News, 8 Nov 2026) on the reliability of
agents that edit business records. Instead of trusting how a run *sounded*,
ThinkingBox executes each task in an isolated backend and grades the **final
state** against a specification.

Microsoft's figures (theirs, as reported):

- Across **121,680 valid trials, 79,853 runs failed** the executable checks —
  and **67.24% of those failures** still ended with a completion-like
  response, a state-changing tool call, and no explicit final tool error.
- The 507-task benchmark repeats every task **20 times**: Claude Opus 5.5
  passed **241 tasks on every attempt**; an unmodified Qwen3.8-27B passed
  only **38 of 507** on every attempt.
- After 50 steps of RL using terminal-state rewards, that Qwen checkpoint's
  all-20 count rose **38 → 107**, and the estimated cost per dependable task
  fell **$24.36 → $8.00** (excluding training).

## What this demo does

Standard library only. A tiny business-records backend (orders, bookings,
permissions), tasks defined as executable checks over final state (goal
fields exact, untouched records identical, a refund policy cap), and a
simulated agent with labelled failure modes: wrong record, partial update
claimed as complete, side effect on an unrelated record, policy breach,
no-op claim, honest partial, visible error, silent crash.

Two graders score every run:

- **Transcript grader** — exactly Microsoft's three misleading signals:
  completion-like final message + at least one state-changing tool call +
  no explicit final tool error.
- **State grader** — runs the executable checks on the final database.

## Results (600 seeded runs, this code)

```
state grader passes:  40.7%
transcript passes:    84.3%
failed runs:          356
of the failures, 266 still looked complete (74.7%)
```

Same shape as Microsoft's 67.24%: the transcript grader waves through
roughly three quarters of the runs that actually broke the records.

## Repeat reliability: where the failures live

24 tasks × 20 attempts per profile, $0.60 per attempt:

```
even         pass@1 0.950 | dependable  8/24 tasks | $36.00 per dependable task
concentrated pass@1 0.931 | dependable 19/24 tasks | $15.16 per dependable task
steady       pass@1 0.900 | dependable  5/24 tasks | $57.60 per dependable task
```

`even` has the best pass@1 and still loses: failing 5% of attempts on
*every* task means almost no task survives 20 repeats. `concentrated` fails
more often overall, but its failures pile up on a hopeless quarter of tasks
while the rest are near-perfect — so it banks more than twice the dependable
tasks at well under half the cost per dependable task. Pass@1 alone ranks
these agents in the wrong order.

## Run it

```
python3 state_grading.py
python3 -m pytest test_state_grading.py -q   # 12 tests
```

## Takeaway for agent builders

Grade state, not transcripts; report pass-all-N alongside pass@1; and price
work per dependable task, not per attempt. A confident final message is a
UI feature, not evidence.
