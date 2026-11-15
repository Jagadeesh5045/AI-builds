# Repetition Guard — Checkpoint Replay for Agent Loops

**Trending context (14 Nov 2026):** Nex-AGI released **Nex-N2.6**, a roughly
321B-parameter multimodal agentic model built on GLM-5.3-Flash under Apache
2.0. Its model card reports 83.0 on OSWorld (vs 78.0 for the base model) and,
the number this demo is about, **reasoning repetition falling from 16.0% to
1.6%** versus Nex-N2.5-Pro on an internal checkpoint-replay evaluation. All
figures in this paragraph are Nex-AGI's own reports, not measurements made
here.

## What this shows

Agents rarely fail by stopping. They fail by *looping*: re-running the same
few actions, convinced each time that this attempt is the one. This demo
builds the defence as an explicit, testable mechanism rather than a model
property:

1. **A simulated agent** works through subtasks (fetch, parse, join, score,
   report, verify). Two subtask kinds are "sticky": on hitting one, the naive
   agent falls into a fixed 3-action cycle (`retry_join, log_wait,
   poll_status`) for 3–6 cycles before progressing.
2. **A repetition detector** (`find_repeated_block`) scans a trace for the
   same action block recurring within a short window — a loop, as opposed to
   the same block reused far apart, which is legitimate.
3. **Checkpoint replay**: progress is snapshotted after each completed
   subtask. When the guard sees the loop precursor on a sticky subtask, it
   rewinds the trace to the checkpoint, adds the looping block to a tabu
   list, and takes the progress action directly.

## Measured results (this code, seeded suite of 40 workloads)

| | Steps | Repeated steps | Repetition rate |
|---|---|---|---|
| Unguarded | 1,251 | 809 | **64.7%** |
| Guarded (checkpoint replay) | 234 | 1 | **0.4%** |

- Loops detected and replayed: **70**
- Budget saved by replaying instead of looping: **1,017 steps (81.3%)**
- Every workload in the suite still completes under the guard.

The rates here are deliberately not comparable to Nex-AGI's 16.0% → 1.6%:
that is a real model on a real benchmark, this is a controlled simulation
where the loop is the dominant failure mode. The shared lesson is the shape
of the fix — detect the repetition early, restore a known-good checkpoint,
and change something (here, a tabu list) so the replay does not walk
straight back into the same cycle.

## Run it

```bash
python3 repetition_guard.py     # prints the guarded vs unguarded suite
python3 -m pytest -q            # 12 tests: detector, agents, suite
```

Standard library only. The detector is independent of the simulation:
`find_repeated_block(trace)` and `count_repeated_steps(trace)` work on any
list of action names, so they can be dropped in front of a real agent's
tool-call log.

## Honest limits

- The guard intervenes on a *known* loop precursor in this simulation; a
  production system would fire purely from the detector on the live stream,
  which costs at least one repeated block before detection.
- Tabu-by-exact-block is brittle against loops that vary one action per
  cycle; fuzzy (edit-distance) block matching is the obvious next step.
- Checkpointing is cheap here because state is a list index. Real agents
  need snapshots of files, tool side effects and context, which is the hard
  part of replay in practice.
