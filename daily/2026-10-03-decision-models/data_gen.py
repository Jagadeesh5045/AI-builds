"""Synthetic agent-telemetry generator.

Each sample is an agent's context vector plus the "correct" action a careful
operator would have taken. The rules encode common-sense behaviour:
- Missing context or ambiguous task -> gather_context
- High risk + low context completeness -> escalate
- Low budget / tools down -> defer
- Otherwise -> proceed

Noise keeps it learnable but not trivially separable.
"""

from __future__ import annotations

import numpy as np

from decision_head import ACTIONS, FEATURE_NAMES

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


def make_dataset(n: int = 4000, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, len(FEATURE_NAMES)))
    risk, context, tools, budget, ambiguity, success = X.T

    y = np.full(n, ACTION_INDEX["proceed"], dtype=int)

    # Rule 1: escalate when risk is high and we lack context or success history
    escalate_mask = (risk > 0.65) & ((context < 0.5) | (success < 0.45))
    # Rule 2: gather_context when the task is ambiguous or info is missing
    gather_mask = ~escalate_mask & ((ambiguity > 0.6) | (context < 0.35))
    # Rule 3: defer when the operating conditions are poor
    defer_mask = ~escalate_mask & ~gather_mask & ((budget < 0.25) | (tools < 0.3))

    y[escalate_mask] = ACTION_INDEX["escalate"]
    y[gather_mask] = ACTION_INDEX["gather_context"]
    y[defer_mask] = ACTION_INDEX["defer"]

    # Label noise: 5% of labels flipped to a random action
    flip = rng.random(n) < 0.05
    y[flip] = rng.integers(0, len(ACTIONS), size=int(flip.sum()))
    return X, y


def example_contexts() -> list[tuple[str, np.ndarray]]:
    """Hand-picked scenarios to show the head's behaviour on demo day."""
    def vec(**kw) -> np.ndarray:
        base = {name: 0.5 for name in FEATURE_NAMES}
        base.update(kw)
        return np.array([base[name] for name in FEATURE_NAMES])

    return [
        ("routine safe task", vec(risk_score=0.1, context_completeness=0.9,
                                  tools_available=1.0, budget_remaining=0.8,
                                  task_ambiguity=0.1, prior_success_rate=0.9)),
        ("vague brief, little context", vec(risk_score=0.2, context_completeness=0.2,
                                            tools_available=0.9, budget_remaining=0.7,
                                            task_ambiguity=0.85, prior_success_rate=0.4)),
        ("risky payment action, thin history", vec(risk_score=0.85, context_completeness=0.4,
                                                   tools_available=0.9, budget_remaining=0.6,
                                                   task_ambiguity=0.4, prior_success_rate=0.3)),
        ("tools down, budget nearly gone", vec(risk_score=0.3, context_completeness=0.7,
                                               tools_available=0.1, budget_remaining=0.05,
                                               task_ambiguity=0.3, prior_success_rate=0.6)),
        # Borderline case: deliberately ambiguous so confidence drops and the
        # gate routes it to a human (the defer-to-human decision-model pattern).
        ("borderline: risky but well-understood", vec(risk_score=0.63, context_completeness=0.56,
                                                      tools_available=0.7, budget_remaining=0.5,
                                                      task_ambiguity=0.5, prior_success_rate=0.5)),
    ]
