# Swarm orchestrator: queen-led multi-agent coding missions

A small, correct, stdlib-only demonstration of the ideas behind **Ruflo**,
announced 13 October 2026 by Cognitum One: an open-source agent
meta-harness / orchestration layer for coding agents. Per the announcement,
Ruflo turns one assistant into a coordinated team of specialized agents with
mission-driven Mods (research, build, test, validate, secure, benchmark,
learn), queen-led swarms, persistent shared memory, routing of each task to
the right model, a run ledger (what ran, what it cost, whether it passed),
and human approval where it matters. (Star/download figures in the launch
coverage are the company's claims, not independently verified here.)

## What this demo shows

`swarm.py` implements the orchestration *mechanism*, with deterministic
toy workers standing in for real specialist models:

1. **Queen-led decomposition.** The `Queen` splits a mission ("fix the
   off-by-one in `top_k`") into Mod-style stages: research, build, test,
   secure (review), benchmark, learn, plus a HIGH-risk deploy step.
2. **Persistent shared memory.** Every worker reads and writes a
   namespaced `SharedMemory` store: the researcher records the bug
   location, the reviewer records its findings, the learner banks a lesson.
3. **Routing to specialists.** Each task goes to the right worker:
   researcher finds the bug, coder writes the fix, tester runs
   deterministic checks, reviewer scans for banned patterns, benchmarker
   compares algorithmic costs, learner distills the lesson.
4. **The review loop.** Revision 1 of the fix is functionally correct (all
   tests pass) but smuggles in an `eval()` helper. The reviewer flags it,
   the coder revises with the findings in memory, and revision 2 ships
   clean. Tests pass both times: the bug was safety, not function.
5. **Approval gate.** The HIGH-risk deploy step asks a human approver.
   Deny it and the mission completes with the step recorded as blocked;
   approve it and the changelog publishes.
6. **Ledger.** Every run is recorded: task, worker, pass/fail, simulated
   tokens, toy cost, gate decision. The demo mission: 10 tasks, 9 passed,
   1 blocked, 4,140 simulated tokens, $0.01242 at the toy rate.

`demo.py` also runs a **naive baseline**: one generalist, no shared memory,
no review, no gate. It ships the `eval()`-tainted fix with zero findings,
which is the whole argument for the harness.

## Run it

```
python demo.py            # mission + denied/approved deploys + naive baseline
python -m unittest -v    # 16 tests
```

## Honest caveats

* This is a **toy proxy**, not Ruflo. The workers are deterministic
  rule-based stand-ins, not language models; tokens and costs are
  simulated, not measured. The takeaway is the *mechanism* (decompose,
  share memory, review, gate, ledger), not the absolute numbers.
* The benchmark compares operation counts of sort-based vs heap-based
  top-k on a synthetic size, not wall-clock time on a real workload.
* Nothing here calls any model API or touches the network.
