# Router/Worker super-agent — a dispatch-only router, isolated workers, and a watchdog

**Trending:** DeepSeek's Harness team open-sourced `dsh-superagent-prototype`
on 31 October 2026 (MIT licence; reported by Zhidx on 1 November, covered in
the 2 November AI dailies). Per that coverage it is a research prototype
validating an architecture, not a commercial product, and some designs are
not fully implemented. Its goal: an always-online agent that runs multiple
tasks in parallel, proactively finds work and recovers from failures, moving
the human role up to decision and acceptance.

## The five pieces, as reported

1. **Dual-layer Router/Worker.** The Router is resident and does global
   routing, human interaction and dispatch only, never heavy execution.
   Workers run isolated sessions with full tools for single tasks and
   report back to the Router.
2. **Event-driven design** — state changes are events, not polling.
3. **Async multi-channel interaction.**
4. **Separation of compute, state and environment.**
5. **An external Watchdog** that watches the workers from outside.

Reported preliminary results (the lab's, from synthetic load in a simplified
environment, not independently verified): p50 task completion 12.4 minutes,
p95 15.7 minutes, better than a single session with subagent tools; Router
batching cut amortised decision-inference time from 6.4s to 1.6s under load.

## What this toy does

`router_worker.py` is a stdlib-only, deterministic, simulated-clock model:

- **Router dispatch split, enforced by a test.** `Router` has no `execute()`
  method and its `executed` counter stays at 0 while workers do all work.
- **Isolated workers.** Each worker has its own scratch state, wiped on a
  watchdog restart; a test proves one worker cannot see another's state.
- **Dependency tickets in parallel waves.** Tickets declare dependencies;
  cycles and unknown dependencies are rejected up front.
- **Failure recovery.** A flaky ticket fails, is requeued by the Router, and
  succeeds on its second attempt.
- **Watchdog.** A worker that stops heartbeating (a hung session) is killed
  from outside, its ticket is requeued with the hang cleared, and a
  replacement worker completes it.
- **Proactive work-finding.** `proactive_scan()` turns TODO markers and
  failing tests in a repo-state dict into tickets nobody requested.
- **Human acceptance gate.** Finished work sits at `awaiting_acceptance`;
  only a human accept marks it accepted, and a reject requeues it.
- **Router batching arithmetic.** One shared decision call over a batch of
  4 reproduces the reported 6.4s → 1.6s amortised effect (`6.4 / batch`).
  Treating that division as valid for larger batches is a toy
  simplification, flagged here rather than hidden.

## Demo output (verified locally, 2 Nov 2026)

```
tickets: 8  workers: 3
parallel makespan: 12 ticks   sequential baseline: 22 ticks
speedup: 1.83x (simulated ticks)
router executed tasks itself: 0 (must be 0)
worker executions: 9
watchdog restarts: 0
router decisions: 9 in 5 batched rounds
amortised decision time: batch of 1 = 6.4s, batch of 4 = 1.6s
accepted: 8/8
events: 39
```

The demo job: scaffold, then API and storage modules in parallel, an
integration step that needs both, a flaky test-fix (fails once, recovers),
docs, plus two proactively scanned tickets. 9 worker executions for 8
tickets is the flaky ticket running twice.

## Tests

14/14 passing: `python3 test_router_worker.py`

Cover: Router-never-executes, worker isolation, dependency ordering,
parallel-beats-sequential, cycle and unknown-dependency detection,
duplicate-ticket rejection, flaky recovery, watchdog restart and
requeue, proactive scanning, human accept/reject, and the batching
arithmetic (including rejecting a batch of zero).

## Honest limits

Simulated ticks are not wall-clock time and the 1.83x is a scheduling
property of this fixture, not a performance claim about any real agent.
No LLM, no network, no real code execution by workers: a worker "doing" a
ticket is modelled as spending its duration and reporting a result, which
is enough to test the routing, isolation, recovery and acceptance
mechanics that are the prototype's actual claim. The p50/p95 and token
figures in the first section are DeepSeek's reported preliminary numbers,
labelled as theirs throughout.
