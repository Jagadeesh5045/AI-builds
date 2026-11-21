"""Looped (weight-shared) transformer blocks, in miniature.

Trending context (21 Nov 2026): Thinking Machines Lab released Inkling 2
Mini on 20 Nov 2026 -- an open-weight (Apache 2.0, as reported) looped
Mixture-of-Experts model for local devices. As reported, it has 24 unique
layers whose middle 12 layers are applied three times per token, so one
token sees 48 layer applications from only 24 layers of weights, with
26.4B total / 3.4B active parameters and a ~20 GB footprint. Those are
the vendor's / coverage's figures, not measurements made here.

This demo isolates the *looping* idea, without MoE or attention:

1. One shared block, applied repeatedly. The block is a fixed message-
   passing operator: each position mixes in its neighbours. Information
   therefore travels exactly one hop per loop, so a task whose answer
   lives ``d`` hops away needs at least ``d`` loops -- depth becomes a
   runtime choice, not a parameter choice.
2. Parameter accounting: a looped stack stores the unique layers once,
   however many times they are applied. We reproduce the reported
   Inkling 2 Mini arithmetic (24 unique -> 48 applications) and the
   general saving versus an unlooped stack of the same effective depth.
3. Adaptive halting (ACT-style): stop looping a position once its state
   stops changing. Easy (near-source) positions halt early; distant
   positions use the full budget -- the looped analogue of spending
   more compute only where the problem is hard.

Pure standard library; states are lists of floats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --------------------------------------------------------------------------
# The shared block: one hop of neighbour mixing
# --------------------------------------------------------------------------

def shared_block(state: list[float], source: list[float]) -> list[float]:
    """One application of the (single, shared) block.

    new[i] = max(state[i], left neighbour, right neighbour, source[i])

    A max (not a mean) over neighbours: an averaging block decays the
    signal by half at every hop, so after a dozen loops nothing is left
    to detect -- the first version of this demo did exactly that and
    its own tests caught it. Max-propagation is the BFS / transitive-
    closure operator: the signal keeps its strength and moves exactly
    one hop per application, which is the property the tests measure.
    The persistent ``source`` term is the residual/input injection:
    the original signal is re-presented at every loop.
    """
    n = len(state)
    out = [0.0] * n
    for i in range(n):
        left = state[i - 1] if i > 0 else 0.0
        right = state[i + 1] if i < n - 1 else 0.0
        out[i] = max(state[i], left, right, source[i])
    return out


@dataclass
class LoopResult:
    state: list[float]
    loops_used: list[int]      # per-position halting loop (adaptive mode)
    total_applications: int    # block applications actually performed


def run_looped(source: list[float], max_loops: int,
               halt_epsilon: float | None = None) -> LoopResult:
    """Apply the shared block up to ``max_loops`` times.

    With ``halt_epsilon=None`` every position is updated every loop
    (fixed-depth looping). With an epsilon, a position halts once its
    own change falls below epsilon and is frozen afterwards; the loop
    still runs while any position is active, but halted positions cost
    no further *effective* applications, which is what we count.
    """
    state = [0.0] * len(source)
    active = [True] * len(source)
    loops_used = [0] * len(source)
    applications = 0
    for _ in range(max_loops):
        if not any(active):
            break
        nxt = shared_block(state, source)
        for i in range(len(state)):
            if not active[i]:
                continue
            change = abs(nxt[i] - state[i])
            state[i] = nxt[i]
            loops_used[i] += 1
            applications += 1
            # Halt only once a position has received signal AND settled.
            # Halting on "no change" alone would freeze distant positions
            # at zero before the signal ever reaches them -- the second
            # bug this demo's tests caught in its first version.
            if (halt_epsilon is not None and state[i] > 0.0
                    and change < halt_epsilon):
                active[i] = False
    return LoopResult(state=state, loops_used=loops_used,
                      total_applications=applications)


def reachable_within(state: list[float], threshold: float = 0.05) -> int:
    """How many positions carry a signal at or above ``threshold``."""
    return sum(1 for v in state if v >= threshold)


# --------------------------------------------------------------------------
# Parameter accounting for looped stacks
# --------------------------------------------------------------------------

@dataclass
class StackShape:
    unique_layers: int
    looped_layers: int    # how many of the unique layers sit in the loop
    loops: int            # applications of each looped layer

    @property
    def applications(self) -> int:
        plain = self.unique_layers - self.looped_layers
        return plain + self.looped_layers * self.loops

    @property
    def unlooped_equivalent_layers(self) -> int:
        """Layers an unlooped stack needs for the same effective depth."""
        return self.applications

    def stored_params(self, params_per_layer: float) -> float:
        return self.unique_layers * params_per_layer

    def unlooped_params(self, params_per_layer: float) -> float:
        return self.unlooped_equivalent_layers * params_per_layer

    def saving_factor(self) -> float:
        return self.unlooped_equivalent_layers / self.unique_layers


def inkling2_mini_shape() -> StackShape:
    """The reported Inkling 2 Mini shape: 24 unique layers, the middle
    12 applied 3 times each -> 6 + 36 + 6 = 48 applications."""
    return StackShape(unique_layers=24, looped_layers=12, loops=3)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

def demo() -> None:
    print("=== Signal propagation: one hop per loop ===")
    n = 15
    source = [0.0] * n
    source[0] = 1.0
    for loops in (1, 2, 4, 7, 15):
        r = run_looped(source, loops)
        print(f"loops={loops:2d}  positions reached (>=0.05): "
              f"{reachable_within(r.state):2d}/{n}  far-end value: "
              f"{r.state[-1]:.3f}")

    print("\n=== Adaptive halting (epsilon=0.01, budget 15) ===")
    fixed = run_looped(source, 15)
    adapt = run_looped(source, 15, halt_epsilon=0.01)
    print(f"fixed   : {fixed.total_applications} position-applications, "
          f"far-end {fixed.state[-1]:.3f}")
    print(f"adaptive: {adapt.total_applications} position-applications, "
          f"far-end {adapt.state[-1]:.3f}")
    print(f"loops used per position: {adapt.loops_used}")
    saved = 1 - adapt.total_applications / fixed.total_applications
    print(f"adaptive saved {saved:.1%} of position-applications at the "
          f"same budget")

    print("\n=== Reported Inkling 2 Mini shape (vendor figures) ===")
    s = inkling2_mini_shape()
    print(f"unique layers {s.unique_layers}, looped {s.looped_layers} "
          f"x{s.loops} -> {s.applications} applications per token")
    print(f"an unlooped stack of the same depth stores "
          f"{s.saving_factor():.2f}x the layer parameters")


if __name__ == "__main__":
    demo()
