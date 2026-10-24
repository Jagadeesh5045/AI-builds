"""Kimi K3.1 Pro-style variant routing and output quality gating.

Toy model only: no API calls, no real model weights.  Pricing and speed
figures are the *reported* Kimi K3.1 Pro numbers (see README.md), used here
as constants so the routing arithmetic can be inspected and tested.

The idea being demonstrated is the one the 23 Oct 2026 launch makes
concrete: when a provider ships the *same* capability as two tiers
(Standard and a 2-3x faster Highspeed tier at roughly double the
per-token price, with cached input priced the same in both), the
deployment question stops being "which model is best?" and becomes
"which tier should *this* task pay for, and did its output pass
automated checks before it ships?"  Moonshot's own account of the
launch stresses larger post-training compute with automated quality
checks, so a quality gate is part of the demo, not an afterthought.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Variant:
    """One deployment tier: price per million tokens, plus speed."""

    name: str
    input_per_m: float       # USD per 1M billable (uncached) input tokens
    output_per_m: float      # USD per 1M output tokens
    cache_per_m: float       # USD per 1M cached input tokens
    speed_factor: float      # 1.0 = Standard speed; 2.5 = mid-point of 2-3x
    base_seconds_per_1k_output: float = 0.6  # toy latency constant


# Reported Kimi pricing (The Decoder, via the launch coverage): Standard
# $2 input / $10 output / $0.20 cached; Highspeed $4 / $20 / $0.20.
STANDARD = Variant("kimi-k3.1-pro", 2.0, 10.0, 0.20, 1.0)
HIGHSPEED = Variant("kimi-k3.1-pro-highspeed", 4.0, 20.0, 0.20, 2.5)


def cost(variant, input_tokens, output_tokens, cached_tokens=0):
    """USD cost of one request. Cached tokens are billed at cache rate."""
    if min(input_tokens, output_tokens, cached_tokens) < 0:
        raise ValueError("token counts must be non-negative")
    if cached_tokens > input_tokens:
        raise ValueError("cached tokens cannot exceed input tokens")
    billable = input_tokens - cached_tokens
    return (
        billable / 1_000_000 * variant.input_per_m
        + cached_tokens / 1_000_000 * variant.cache_per_m
        + output_tokens / 1_000_000 * variant.output_per_m
    )


def latency_seconds(variant, input_tokens, output_tokens):
    """Toy latency: output-bound, with a small long-context scan term."""
    raw = (
        output_tokens / 1000 * variant.base_seconds_per_1k_output
        + input_tokens / 10_000 * 0.02
    )
    return raw / variant.speed_factor


@dataclass
class Task:
    kind: str  # "coding", "finance", "cad" or "agent"
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    urgency: float = 0.0  # 0 = overnight batch, 1 = interactive user waiting
    deadline_seconds: float = 600.0


@dataclass
class Decision:
    variant: Variant
    reason: str


def route(task):
    """Pick a tier for one task. Rules are explicit, in priority order.

    1. Deadline: if Standard cannot meet the deadline but Highspeed can,
       pay the premium (a missed deadline is worse than a higher bill).
    2. Interactivity: urgency >= 0.8 always pays for speed.
    3. Cache: when over half the context is cached, the Highspeed
       premium shrinks (cached input costs the same in both tiers), so
       moderate urgency (>= 0.3) is enough to justify Highspeed.
    4. Otherwise Standard: batch work should not pay a latency premium.
    """
    std_lat = latency_seconds(STANDARD, task.input_tokens, task.output_tokens)
    hs_lat = latency_seconds(HIGHSPEED, task.input_tokens, task.output_tokens)
    if std_lat > task.deadline_seconds >= hs_lat:
        return Decision(HIGHSPEED,
                        f"deadline: standard {std_lat:.2f}s > "
                        f"{task.deadline_seconds:.2f}s, highspeed {hs_lat:.2f}s")
    if task.urgency >= 0.8:
        return Decision(HIGHSPEED, f"interactive: urgency {task.urgency:.2f}")
    cached_share = task.cached_tokens / task.input_tokens if task.input_tokens else 0.0
    if cached_share >= 0.5 and task.urgency >= 0.3:
        return Decision(HIGHSPEED,
                        f"cache-heavy: {cached_share:.0%} cached, "
                        "premium is small, urgency moderate")
    return Decision(STANDARD, "batch/default: no deadline or urgency pressure")


# --- Automated quality gate (the post-training-check idea, at serving time) ---

REQUIRED_FIELDS = {
    "coding": ("language", "code"),
    "finance": ("ticker", "total"),
    "cad": ("units", "dimensions"),
    "agent": ("action", "target"),
}
PLACEHOLDERS = ("todo", "tbd", "lorem", "xxx")


@dataclass
class ModelOutput:
    kind: str
    text: str
    fields: dict = field(default_factory=dict)
    # Each numeric check is (label, expected, actual, tolerance).
    numeric_checks: list = field(default_factory=list)


@dataclass
class GateResult:
    passed: bool
    reasons: list


def check_output(output):
    """Return a GateResult; an output passes only with no reasons to fail."""
    reasons = []
    for name in REQUIRED_FIELDS.get(output.kind, ()):
        value = output.fields.get(name)
        if not value:
            reasons.append(f"missing required field: {name}")
    lowered = output.text.lower()
    for token in PLACEHOLDERS:
        if token in lowered:
            reasons.append(f"placeholder present: {token}")
    for label, expected, actual, tolerance in output.numeric_checks:
        if abs(actual - expected) > tolerance:
            reasons.append(
                f"numeric mismatch: {label} expected {expected}, got {actual}")
    return GateResult(passed=not reasons, reasons=reasons)


# --- The per-token price vs per-task cost lesson ---------------------------

def generation_cost_factor(old_output_price, new_output_price, output_growth):
    """Effective output-cost multiplier after a price cut + more tokens.

    Kimi K3 -> K3.1 Pro: output price $15 -> $10 per million, but the new
    model reportedly generates ~14% more output tokens, so the multiplier
    is (10/15) * 1.14 = 0.76, not 0.67. A lower sticker price still saves
    money here, but the saving shrinks as output share grows -- which is
    why the reported Artificial Analysis task cost barely moved
    ($2.00 -> $1.98) despite a 33% per-token cut.
    """
    return (new_output_price / old_output_price) * (1.0 + output_growth)


def compare_k3_k31(input_tokens, output_tokens, cached_tokens=0):
    """Cost of the same task on K3 vs K3.1 Pro, with 14% output growth."""
    k3 = Variant("kimi-k3", 3.0, 15.0, 0.30, 1.0)
    grown_output = round(output_tokens * 1.14)
    return {
        "k3_cost": cost(k3, input_tokens, output_tokens, cached_tokens),
        "k31_cost": cost(STANDARD, input_tokens, grown_output, cached_tokens),
        "k31_output_tokens": grown_output,
    }
