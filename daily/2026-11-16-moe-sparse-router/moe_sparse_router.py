"""
Sparse Mixture-of-Experts (MoE) router demo.

Trending context (16 Nov 2026): Alibaba released Qwen4-Max, a
Mixture-of-Experts flagship reported at 2.4 trillion total parameters
activating only about 71 billion per token, with a 1M-token context.
Those figures, and the Work360 cost figures used below, are Alibaba's
own published claims, not independently verified here.

This module shows, in pure standard-library Python, the three ideas
that make sparse MoE work in practice:

1. Top-k routing: a small router scores every expert for each token
   and only the top-k experts run, so compute per token tracks the
   ACTIVE parameter count, not the total.
2. Load balancing: without a penalty, a router collapses onto a few
   popular experts. The Switch-style auxiliary loss used here is
   ``E * sum_i(mean_prob_i * fraction_i)``; it equals 1.0 for a
   perfectly balanced router and grows with imbalance.
3. Expert capacity: each expert accepts a bounded number of tokens
   per batch (capacity factor). Overflow tokens are dropped to the
   residual path and counted, never silently processed twice.

It also provides the active-vs-total accounting a serving team
actually budgets with: active fraction, a FLOPs proxy per token and
a cost-per-task estimate from a price per million active tokens.

No third-party packages. Vectors are plain lists of floats.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# Alibaba's own reported Qwen4-Max figures (16 Nov 2026 launch
# coverage, from Alibaba Cloud documentation and its Work360 test).
# Labelled as the vendor's claims wherever they are used.
QWEN4_MAX_TOTAL_PARAMS_B = 2400.0
QWEN4_MAX_ACTIVE_PARAMS_B = 71.0
QWEN4_MAX_WORK360_COST_BEFORE = 4.43   # USD per task, Qwen3.8-Max
QWEN4_MAX_WORK360_COST_AFTER = 1.67    # USD per task, Qwen4-Max


def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax."""
    if not logits:
        raise ValueError("logits must not be empty")
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class Expert:
    """A tiny two-layer MLP expert with deterministic seeded weights."""

    name: str
    dim: int
    hidden: int
    seed: int
    w1: list[list[float]] = field(default_factory=list)
    w2: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        scale1 = 1.0 / math.sqrt(self.dim)
        scale2 = 1.0 / math.sqrt(self.hidden)
        self.w1 = [[rng.uniform(-scale1, scale1) for _ in range(self.dim)]
                   for _ in range(self.hidden)]
        self.w2 = [[rng.uniform(-scale2, scale2) for _ in range(self.hidden)]
                   for _ in range(self.dim)]

    @property
    def param_count(self) -> int:
        return self.hidden * self.dim + self.dim * self.hidden

    def __call__(self, x: list[float]) -> list[float]:
        if len(x) != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {len(x)}")
        hidden = [max(0.0, sum(w * v for w, v in zip(row, x)))
                  for row in self.w1]
        return [sum(w * h for w, h in zip(row, hidden)) for row in self.w2]


@dataclass
class Router:
    """Linear router: projects a token to one logit per expert."""

    dim: int
    num_experts: int
    seed: int = 7
    weights: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        scale = 1.0 / math.sqrt(self.dim)
        self.weights = [[rng.uniform(-scale, scale) for _ in range(self.dim)]
                        for _ in range(self.num_experts)]

    def logits(self, x: list[float]) -> list[float]:
        return [sum(w * v for w, v in zip(row, x)) for row in self.weights]

    def route(self, x: list[float], top_k: int) -> tuple[list[int], list[float], list[float]]:
        """Return (expert indices, renormalised weights, full probs)."""
        if not 1 <= top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        probs = softmax(self.logits(x))
        order = sorted(range(self.num_experts), key=lambda i: -probs[i])
        chosen = order[:top_k]
        picked = [probs[i] for i in chosen]
        total = sum(picked)
        weights = [p / total for p in picked] if total > 0 else [1.0 / top_k] * top_k
        return chosen, weights, probs


def load_balancing_loss(all_probs: list[list[float]],
                        assignments: list[list[int]],
                        num_experts: int) -> float:
    """Switch-style auxiliary loss.

    ``E * sum_i(mean_prob_i * fraction_i)`` where ``fraction_i`` is
    the share of routing slots given to expert ``i``. Balanced = 1.0.
    """
    if not all_probs or not assignments:
        raise ValueError("need at least one token")
    n = len(all_probs)
    mean_prob = [sum(p[i] for p in all_probs) / n for i in range(num_experts)]
    slots = sum(len(a) for a in assignments)
    counts = [0] * num_experts
    for chosen in assignments:
        for idx in chosen:
            counts[idx] += 1
    fraction = [c / slots for c in counts]
    return num_experts * sum(mp * fr for mp, fr in zip(mean_prob, fraction))


@dataclass
class MoEResult:
    outputs: list[list[float]]
    expert_load: list[int]        # tokens actually processed per expert
    dropped: int                  # tokens dropped by capacity, per slot
    aux_loss: float
    mean_active_fraction: float   # active params / total params


class SparseMoELayer:
    """Top-k sparse MoE layer with per-expert capacity."""

    def __init__(self, dim: int = 16, hidden: int = 32, num_experts: int = 8,
                 top_k: int = 2, capacity_factor: float = 1.25, seed: int = 7):
        if capacity_factor < 1.0:
            raise ValueError("capacity_factor must be >= 1.0")
        self.dim, self.top_k = dim, top_k
        self.capacity_factor = capacity_factor
        self.router = Router(dim=dim, num_experts=num_experts, seed=seed)
        self.experts = [Expert(name=f"expert_{i}", dim=dim, hidden=hidden,
                               seed=seed + 100 + i)
                        for i in range(num_experts)]

    @property
    def total_params(self) -> int:
        router_params = self.router.num_experts * self.dim
        return sum(e.param_count for e in self.experts) + router_params

    @property
    def active_params(self) -> int:
        """Router plus the top-k experts that run for one token."""
        router_params = self.router.num_experts * self.dim
        return self.top_k * self.experts[0].param_count + router_params

    def capacity(self, num_tokens: int) -> int:
        per_expert = num_tokens * self.top_k / len(self.experts)
        return max(1, math.ceil(self.capacity_factor * per_expert))

    def __call__(self, tokens: list[list[float]]) -> MoEResult:
        if not tokens:
            raise ValueError("need at least one token")
        cap = self.capacity(len(tokens))
        load = [0] * len(self.experts)
        dropped = 0
        outputs: list[list[float]] = []
        all_probs: list[list[float]] = []
        assignments: list[list[int]] = []
        for x in tokens:
            chosen, weights, probs = self.router.route(x, self.top_k)
            all_probs.append(probs)
            assignments.append(chosen)
            out = list(x)  # residual path
            for idx, weight in zip(chosen, weights):
                if load[idx] >= cap:
                    dropped += 1
                    continue
                load[idx] += 1
                expert_out = self.experts[idx](x)
                out = [o + weight * e for o, e in zip(out, expert_out)]
            outputs.append(out)
        aux = load_balancing_loss(all_probs, assignments, len(self.experts))
        return MoEResult(outputs=outputs, expert_load=load, dropped=dropped,
                         aux_loss=aux,
                         mean_active_fraction=self.active_params / self.total_params)


def active_fraction(total_params_b: float, active_params_b: float) -> float:
    """Share of parameters that run per token."""
    if total_params_b <= 0:
        raise ValueError("total params must be positive")
    if not 0 < active_params_b <= total_params_b:
        raise ValueError("active params must be in (0, total]")
    return active_params_b / total_params_b


def flops_proxy_per_token(active_params_b: float) -> float:
    """Rough forward-pass FLOPs proxy: about 2 FLOPs per active param."""
    return 2.0 * active_params_b * 1e9


def cost_per_task(tokens_per_task: float, price_per_million_active_tokens: float,
                  active_params_b: float, reference_active_params_b: float) -> float:
    """Cost scaled by active size relative to a priced reference model.

    A model that activates fewer parameters per token is cheaper to
    serve at the same token count, all else equal. This is a planning
    proxy, not a billing formula.
    """
    if reference_active_params_b <= 0:
        raise ValueError("reference active params must be positive")
    token_cost = tokens_per_task / 1_000_000 * price_per_million_active_tokens
    return token_cost * (active_params_b / reference_active_params_b)


def make_tokens(n: int, dim: int, seed: int = 42) -> list[list[float]]:
    """Seeded synthetic tokens clustered into a few topic groups."""
    rng = random.Random(seed)
    centres = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(4)]
    tokens = []
    for i in range(n):
        centre = centres[i % len(centres)]
        tokens.append([c + rng.gauss(0, 0.3) for c in centre])
    return tokens


def _demo() -> None:
    layer = SparseMoELayer()
    tokens = make_tokens(512, layer.dim)
    result = layer(tokens)
    mean_load = sum(result.expert_load) / len(result.expert_load)
    imbalance = max(result.expert_load) / mean_load if mean_load else 0.0
    frac = active_fraction(QWEN4_MAX_TOTAL_PARAMS_B, QWEN4_MAX_ACTIVE_PARAMS_B)
    saving = 1 - QWEN4_MAX_WORK360_COST_AFTER / QWEN4_MAX_WORK360_COST_BEFORE
    print("Sparse MoE demo: 512 tokens, 8 experts, top-2 routing")
    print(f"  expert load:        {result.expert_load}")
    print(f"  load imbalance:     max/mean = {imbalance:.2f}")
    print(f"  dropped slots:      {result.dropped}")
    print(f"  aux balancing loss: {result.aux_loss:.3f} (1.000 = balanced)")
    print(f"  demo layer active:  {result.mean_active_fraction:.1%} of params per token")
    print(f"  Qwen4-Max reported: {QWEN4_MAX_ACTIVE_PARAMS_B:.0f}B active of "
          f"{QWEN4_MAX_TOTAL_PARAMS_B:.0f}B total = {frac:.1%} per token")
    print(f"  Work360 cost (Alibaba's own figures): "
          f"${QWEN4_MAX_WORK360_COST_BEFORE:.2f} -> ${QWEN4_MAX_WORK360_COST_AFTER:.2f} "
          f"per task ({saving:.0%} lower)")


if __name__ == "__main__":
    _demo()
