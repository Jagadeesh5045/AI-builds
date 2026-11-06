"""
Adaptive reasoning effort: spend compute where the question needs it.

Trending context (5 Nov 2026): Thinking Machines Lab released Inkling 2
Small, a 594B-parameter sparse MoE with 17B active parameters whose model
card highlights *variable reasoning effort* -- the model can spend more
or less reasoning compute per request. This demo isolates that one idea
in pure Python, with no model at all:

  * a "reasoning sample" is a noisy walk through a multi-step arithmetic
    chain (each step can slip, like a reasoning trace can slip);
  * self-consistency voting turns several samples into one answer, and
    the agreement fraction is a cheap confidence signal;
  * an effort controller starts with 3 samples and only buys more
    (5, then 9) when the samples disagree, mirroring how a variable-
    effort model should allocate its reasoning budget.

Three strategies are compared on the same seeded problem set:

  fixed_low   1 sample, no vote          (cheapest, weakest)
  fixed_high  9 samples, always          (strongest, wasteful)
  adaptive    3 -> 5 -> 9, stops early when the vote is convincing

Run:  python3 adaptive_reasoning.py
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

STEP_ERROR_P = 0.07          # chance one step in one sample slips
TOKEN_PER_STEP = 12          # nominal reasoning tokens per step


# ---------------------------------------------------------------- problems

@dataclass
class Problem:
    start: int
    ops: list[tuple[str, int]]          # ("+", 4), ("*", 3), ...
    answer: int

    @property
    def steps(self) -> int:
        return len(self.ops)


def apply_op(value: int, op: str, operand: int) -> int:
    if op == "+":
        return value + operand
    if op == "-":
        return value - operand
    if op == "*":
        return value * operand
    raise ValueError(op)


def make_problem(rng: random.Random, steps: int) -> Problem:
    """A chain computation with an exactly known answer."""
    value = rng.randint(2, 9)
    start = value
    ops: list[tuple[str, int]] = []
    for _ in range(steps):
        op = rng.choice(["+", "+", "-", "*"])
        operand = rng.randint(2, 9) if op == "*" else rng.randint(3, 40)
        value = apply_op(value, op, operand)
        ops.append((op, operand))
    return Problem(start=start, ops=ops, answer=value)


def make_problem_set(n: int, seed: int) -> list[Problem]:
    rng = random.Random(seed)
    return [make_problem(rng, steps=rng.randint(2, 7)) for _ in range(n)]


# ------------------------------------------------------- one noisy sample

def reasoning_sample(problem: Problem, rng: random.Random) -> int:
    """Walk the chain; each step slips with probability STEP_ERROR_P.

    A slip perturbs the running value by a small non-zero amount, so a
    wrong trace lands on a wrong (and usually distinct) answer -- the
    same property that makes majority voting over traces useful.
    """
    value = problem.start
    for op, operand in problem.ops:
        value = apply_op(value, op, operand)
        if rng.random() < STEP_ERROR_P:
            value += rng.choice([-1, 1]) * rng.randint(1, 9)
    return value


# ------------------------------------------------------------- voting

@dataclass
class Vote:
    answer: int
    confidence: float            # share of samples backing the winner
    samples: int


def majority_vote(answers: list[int]) -> Vote:
    counts = Counter(answers)
    answer, top = counts.most_common(1)[0]
    return Vote(answer=answer, confidence=top / len(answers),
                samples=len(answers))


# ------------------------------------------------------- effort control

@dataclass
class EffortResult:
    answer: int
    confidence: float
    samples_used: int
    tokens_used: int


def tokens_for(problem: Problem, samples: int) -> int:
    return samples * problem.steps * TOKEN_PER_STEP


def solve_fixed(problem: Problem, samples: int,
                rng: random.Random) -> EffortResult:
    answers = [reasoning_sample(problem, rng) for _ in range(samples)]
    vote = majority_vote(answers)
    return EffortResult(vote.answer, vote.confidence, samples,
                        tokens_for(problem, samples))


def solve_adaptive(problem: Problem, rng: random.Random,
                   ladder: tuple[int, ...] = (3, 5, 9),
                   stop_confidence: float = 0.8) -> EffortResult:
    """Buy samples in stages; stop as soon as the vote is convincing.

    Stage sizes are cumulative targets on the ladder. A unanimous or
    near-unanimous vote (confidence >= stop_confidence, with at least 3
    samples behind it) ends the spend; disagreement escalates effort.
    """
    answers: list[int] = []
    vote = majority_vote([reasoning_sample(problem, rng)])
    for target in ladder:
        while len(answers) < target:
            answers.append(reasoning_sample(problem, rng))
        vote = majority_vote(answers)
        if vote.confidence >= stop_confidence:
            break
    return EffortResult(vote.answer, vote.confidence, len(answers),
                        tokens_for(problem, len(answers)))


# ------------------------------------------------------------ benchmark

@dataclass
class StrategyStats:
    name: str
    correct: int = 0
    total: int = 0
    samples: int = 0
    tokens: int = 0
    by_steps: dict[int, list[int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def avg_samples(self) -> float:
        return self.samples / self.total if self.total else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.tokens / self.total if self.total else 0.0


def run_benchmark(n: int = 300, seed: int = 20261106) -> dict[str, StrategyStats]:
    problems = make_problem_set(n, seed)
    strategies = {
        "fixed_low": lambda p, r: solve_fixed(p, 1, r),
        "fixed_high": lambda p, r: solve_fixed(p, 9, r),
        "adaptive": lambda p, r: solve_adaptive(p, r),
    }
    out: dict[str, StrategyStats] = {}
    for name, solver in strategies.items():
        # Same per-problem RNG stream per strategy: paired comparison.
        stats = StrategyStats(name)
        for i, problem in enumerate(problems):
            rng = random.Random(seed * 1000 + i)
            res = solver(problem, rng)
            stats.total += 1
            stats.correct += int(res.answer == problem.answer)
            stats.samples += res.samples_used
            stats.tokens += res.tokens_used
            bucket = stats.by_steps.setdefault(problem.steps, [0, 0])
            bucket[1] += 1
            bucket[0] += int(res.answer == problem.answer)
        out[name] = stats
    return out


def main() -> None:
    results = run_benchmark()
    print("Adaptive reasoning effort - self-consistency budget control")
    print("300 seeded chain problems, 2-7 steps each, "
          f"per-step slip probability {STEP_ERROR_P}\n")
    print(f"{'strategy':<12} {'accuracy':>9} {'avg samples':>12} "
          f"{'avg tokens':>11}")
    for name in ("fixed_low", "fixed_high", "adaptive"):
        s = results[name]
        print(f"{name:<12} {s.accuracy:>9.3f} {s.avg_samples:>12.2f} "
              f"{s.avg_tokens:>11.1f}")
    ad, hi = results["adaptive"], results["fixed_high"]
    saved = 1 - ad.avg_samples / hi.avg_samples
    print(f"\nAdaptive keeps {ad.accuracy:.3f} accuracy vs fixed-high "
          f"{hi.accuracy:.3f} while using {saved:.0%} fewer samples.")
    print("\nAdaptive accuracy by problem length (steps):")
    for steps in sorted(ad.by_steps):
        c, t = ad.by_steps[steps]
        print(f"  {steps} steps: {c}/{t} = {c / t:.3f}")


if __name__ == "__main__":
    main()
