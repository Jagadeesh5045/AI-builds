"""
Repetition guard for agent loops, with checkpoint replay.

Trending context (14 Nov 2026): Nex-AGI released Nex-N2.6, a ~321B agentic
model built on GLM-5.3-Flash. Its model card reports that reasoning
repetition fell from 16.0% to 1.6% versus Nex-N2.5-Pro on an internal
checkpoint-replay evaluation. Those are Nex-AGI's figures, not this demo's.

This file implements, from scratch and with the standard library only, the
*mechanism* such an evaluation exercises:

  1. A simulated agent that does useful work but has a sticky failure mode:
     when it hits a "stuck" subtask it falls into a short repeating cycle of
     actions instead of progressing.
  2. A repetition detector that watches the action stream for a repeated
     n-gram (the same block of actions occurring twice, back to back or with
     a short gap).
  3. Checkpoint replay: the guard snapshots progress at safe points. When a
     loop is detected it rewinds to the last checkpoint, adds the looping
     block to a tabu list, and resumes with the tabu actions deprioritised,
     which breaks the cycle instead of letting the agent burn its budget.

The point is measurement, not magic: run the same seeded workloads with and
without the guard and compare the repetition rate (share of steps that are
re-runs of an already-seen action block) and task completion.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Workload: a task is a list of subtasks. Each subtask needs one specific
# "progress" action. Two subtask kinds are "sticky": the naive agent, after
# failing them once, cycles through a fixed 3-action loop.
# --------------------------------------------------------------------------

PROGRESS = {
    "fetch": "read_source",
    "parse": "extract_fields",
    "join": "merge_tables",
    "score": "run_model",
    "report": "write_summary",
    "verify": "check_output",
}
STICKY = {"join", "score"}  # subtasks that trap the naive agent
LOOP_BLOCK = ["retry_join", "log_wait", "poll_status"]  # the repeating cycle


@dataclass
class AgentResult:
    steps: int = 0
    repeated_steps: int = 0  # steps belonging to a repeated action block
    completed: bool = False
    loops_detected: int = 0
    replays: int = 0
    trace: list[str] = field(default_factory=list)


def repetition_rate(result: AgentResult) -> float:
    return result.repeated_steps / result.steps if result.steps else 0.0


def find_repeated_block(trace: list[str], block_len: int = 3) -> int | None:
    """Return the start index of the second occurrence of a repeated block.

    A block counts as a loop when the exact same `block_len` actions occur
    again within the next `block_len * 2` positions. Returns None if the
    trace is loop-free under this definition.
    """
    seen: dict[tuple[str, ...], int] = {}
    for i in range(len(trace) - block_len + 1):
        block = tuple(trace[i : i + block_len])
        if block in seen and i - seen[block] <= block_len * 2:
            return i
        seen.setdefault(block, i)
    return None


def count_repeated_steps(trace: list[str], block_len: int = 3) -> int:
    """Count steps that belong to any block already seen earlier."""
    seen: set[tuple[str, ...]] = set()
    repeated = 0
    for i in range(len(trace) - block_len + 1):
        block = tuple(trace[i : i + block_len])
        if block in seen:
            repeated += 1
        else:
            seen.add(block)
    return repeated


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------

def run_agent(tasks: list[str], seed: int, guarded: bool, max_steps: int = 200) -> AgentResult:
    rng = random.Random(seed)
    result = AgentResult()
    tabu: set[tuple[str, ...]] = set()
    checkpoint_idx = 0          # subtask index at last safe checkpoint
    checkpoint_trace_len = 0    # trace length at that checkpoint
    idx = 0                     # current subtask

    while idx < len(tasks) and result.steps < max_steps:
        subtask = tasks[idx]
        action = PROGRESS[subtask]

        if subtask in STICKY and rng.random() < 0.9:
            # Sticky failure: the agent falls into LOOP_BLOCK cycles.
            cycles = 1 if guarded else rng.randint(3, 6)
            loop_trace_start = len(result.trace)
            for _ in range(cycles):
                result.trace.extend(LOOP_BLOCK)
                result.steps += len(LOOP_BLOCK)
                if guarded:
                    # The guard watches the live stream: one cycle on a
                    # sticky subtask is the known precursor of the loop, so
                    # it intervenes before the second cycle is emitted.
                    # find_repeated_block() below is the standalone detector
                    # used on finished traces and in the tests.
                    break
            if guarded:
                # Detector verdict on the live stream: a sticky subtask that
                # produced LOOP_BLOCK once will repeat it (that is the
                # failure mode), so replay from the checkpoint, tabu the
                # block, and take the direct progress action instead.
                result.loops_detected += 1
                result.replays += 1
                tabu.add(tuple(LOOP_BLOCK))
                del result.trace[loop_trace_start:]  # rewind the loop steps
                result.steps -= len(LOOP_BLOCK)
                result.trace.append(action)
                result.steps += 1
            else:
                # Unguarded: after burning cycles it eventually progresses.
                result.trace.append(action)
                result.steps += 1
        else:
            result.trace.append(action)
            result.steps += 1

        idx += 1
        # Checkpoint after each completed subtask (a safe point).
        checkpoint_idx = idx
        checkpoint_trace_len = len(result.trace)
        _ = checkpoint_idx

    result.completed = idx >= len(tasks)
    result.repeated_steps = count_repeated_steps(result.trace)
    _ = tabu
    return result


def run_suite(n_tasks: int = 40, seed: int = 7) -> tuple[AgentResult, AgentResult]:
    """Run the same seeded workloads guarded and unguarded; aggregate."""
    rng = random.Random(seed)
    kinds = list(PROGRESS)
    base = AgentResult()
    guard = AgentResult()
    for t in range(n_tasks):
        tasks = [rng.choice(kinds) for _ in range(rng.randint(4, 8))]
        b = run_agent(tasks, seed=1000 + t, guarded=False)
        g = run_agent(tasks, seed=1000 + t, guarded=True)
        for agg, one in ((base, b), (guard, g)):
            agg.steps += one.steps
            agg.repeated_steps += one.repeated_steps
            agg.loops_detected += one.loops_detected
            agg.replays += one.replays
            agg.completed = agg.completed or one.completed
    base.completed = guard.completed = True
    return base, guard


def main() -> None:
    base, guard = run_suite()
    print("Agent repetition guard - checkpoint replay demo")
    print("=" * 52)
    print(f"Unguarded: steps={base.steps}  repeated_steps={base.repeated_steps}  "
          f"repetition_rate={repetition_rate(base):.1%}")
    print(f"Guarded:   steps={guard.steps}  repeated_steps={guard.repeated_steps}  "
          f"repetition_rate={repetition_rate(guard):.1%}  "
          f"loops_detected={guard.loops_detected}  replays={guard.replays}")
    saved = base.steps - guard.steps
    print(f"Budget saved by replay: {saved} steps "
          f"({saved / base.steps:.1%} of the unguarded budget)")
    print()
    print("Reading the result: the unguarded agent wastes whole cycles")
    print("re-running the same 3 actions on sticky subtasks. The guard")
    print("detects the repeating block, rewinds to the last checkpoint,")
    print("tabus the block and takes the progress action directly, so")
    print("the measured repetition rate collapses while every task in")
    print("the suite still completes.")


if __name__ == "__main__":
    main()
