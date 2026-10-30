"""Agent release-health monitor: catch regressions without an LLM judge.

Inspired by Oimo AI's Apgar (open-sourced 29 October 2026), which watches
Langfuse production traces and compares releases on observable metrics
instead of paying for an LLM judge over sampled traffic.

Idea
----
A release is a set of traces. Each trace carries its release in its
metadata, so releases are detected automatically -- no manual tagging.
Two releases are compared on three observable surfaces only:

1. success rate (two-proportion z-test, so a drop is not just noise),
2. latency (mean, p50, p95 -- a slow release can be a broken release),
3. error mix (a brand-new error type, or a sharp rise in an old one).

Why full traffic, not a sampled judge? A regression that lives in one
route (here: tool calls) is diluted in a small sample: in the demo the
sampled success-rate drop is statistically inconclusive (p=0.29),
while the full-traffic comparison puts it at p=0.0001. A judge that
only samples can therefore waver on the headline metric even when the
regression is real; full-traffic observable metrics do not.

Standard library only. No API keys, no network, no LLM calls.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

MIN_SAMPLE = 50          # below this, a verdict is not trustworthy
SUCCESS_DROP_PP = 0.03   # success-rate fall that counts as a regression
ERROR_RISE_PP = 0.05     # error-type share rise that counts as a regression
P95_RISE = 0.20          # p95 latency rise that counts (with mean rise)
MEAN_RISE = 0.10
SIGNIFICANCE = 0.05


# ---------------------------------------------------------------- traces

def group_traces(traces):
    """Group traces by the release recorded in their metadata.

    Returns {release: [trace, ...]}. Traces without a release are ignored,
    because an untagged trace cannot be attributed to a deployment.
    """
    grouped = defaultdict(list)
    for trace in traces:
        release = trace.get("release")
        if release:
            grouped[release].append(trace)
    return dict(grouped)


# ------------------------------------------------------------- statistics

def success_rate(traces):
    """Fraction of traces that succeeded (0.0 for an empty list)."""
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.get("success")) / len(traces)


def percentile(values, pct):
    """Nearest-rank percentile of a non-empty list of numbers."""
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(values)
    rank = math.ceil(pct / 100.0 * len(ordered))
    return ordered[max(0, min(rank - 1, len(ordered) - 1))]


def _phi(z):
    """Standard normal CDF, via the error function (stdlib only)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_proportion_pvalue(success_a, n_a, success_b, n_b):
    """Two-sided p-value for a difference in two success proportions.

    Returns 1.0 when either sample is empty or the pooled variance is 0
    (identical proportions carry no evidence of a difference).
    """
    if n_a == 0 or n_b == 0:
        return 1.0
    p_a, p_b = success_a / n_a, success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    var = pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b)
    if var <= 0:
        return 1.0
    z = (p_a - p_b) / math.sqrt(var)
    return 2.0 * (1.0 - _phi(abs(z)))


def error_mix(traces):
    """Share of traces per error type (failed traces only)."""
    failures = [t for t in traces if not t.get("success")]
    if not failures:
        return {}
    counts = Counter(t.get("error") or "unknown" for t in failures)
    return {err: count / len(traces) for err, count in counts.items()}


def latency_summary(traces):
    """Mean / p50 / p95 latency in milliseconds."""
    latencies = [t["latency_ms"] for t in traces if "latency_ms" in t]
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": sum(latencies) / len(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
    }


# ------------------------------------------------------------- comparison

def compare_releases(baseline, candidate):
    """Compare a candidate release against a baseline release.

    Returns a dict with the measured differences and a verdict:
    "REGRESSION", "NO REGRESSION", "WATCH" (moved, but not past a
    threshold) or "INCONCLUSIVE" (too few traces to judge).
    """
    n_base, n_cand = len(baseline), len(candidate)
    if n_base < MIN_SAMPLE or n_cand < MIN_SAMPLE:
        return {
            "verdict": "INCONCLUSIVE",
            "reasons": [f"sample too small (baseline={n_base}, candidate={n_cand}, need >={MIN_SAMPLE})"],
            "n_baseline": n_base, "n_candidate": n_cand,
        }

    succ_base = sum(1 for t in baseline if t.get("success"))
    succ_cand = sum(1 for t in candidate if t.get("success"))
    rate_base, rate_cand = succ_base / n_base, succ_cand / n_cand
    p_value = two_proportion_pvalue(succ_base, n_base, succ_cand, n_cand)
    lat_base, lat_cand = latency_summary(baseline), latency_summary(candidate)
    err_base, err_cand = error_mix(baseline), error_mix(candidate)

    reasons = []
    if rate_cand - rate_base <= -SUCCESS_DROP_PP and p_value < SIGNIFICANCE:
        reasons.append(
            f"success rate fell {rate_base:.3f} -> {rate_cand:.3f} "
            f"({(rate_base - rate_cand) * 100:.1f}pp, p={p_value:.4f})"
        )
    mean_rise = (lat_cand["mean"] - lat_base["mean"]) / lat_base["mean"] if lat_base["mean"] else 0.0
    p95_rise = (lat_cand["p95"] - lat_base["p95"]) / lat_base["p95"] if lat_base["p95"] else 0.0
    if p95_rise >= P95_RISE and mean_rise >= MEAN_RISE:
        reasons.append(
            f"latency rose: mean {lat_base['mean']:.0f} -> {lat_cand['mean']:.0f}ms, "
            f"p95 {lat_base['p95']:.0f} -> {lat_cand['p95']:.0f}ms"
        )
    for err, share in err_cand.items():
        rise = share - err_base.get(err, 0.0)
        if rise >= ERROR_RISE_PP:
            reasons.append(f"error '{err}' share rose {err_base.get(err, 0.0):.3f} -> {share:.3f}")

    if reasons:
        verdict = "REGRESSION"
    elif rate_cand < rate_base or mean_rise > 0:
        verdict = "WATCH"
    else:
        verdict = "NO REGRESSION"

    return {
        "verdict": verdict,
        "reasons": reasons,
        "n_baseline": n_base,
        "n_candidate": n_cand,
        "success_baseline": rate_base,
        "success_candidate": rate_cand,
        "success_p_value": p_value,
        "latency_baseline": lat_base,
        "latency_candidate": lat_cand,
        "error_baseline": err_base,
        "error_candidate": err_cand,
    }


# ------------------------------------------------------- synthetic traces

def generate_traces(release, n, rng, degraded_tool_route=False):
    """Deterministic synthetic Langfuse-style traces for one release.

    Routes: chat (55%), tool_call (25%), retrieval (20%). In the degraded
    variant the tool_call route drops from 96% to 70% success and its
    failures are timeouts with slow latencies -- the kind of route-local
    regression a sampled judge can miss.
    """
    routes = ["chat", "tool_call", "retrieval"]
    weights = [0.55, 0.25, 0.20]
    base_success = {"chat": 0.95, "tool_call": 0.96, "retrieval": 0.92}
    traces = []
    for i in range(n):
        route = rng.choices(routes, weights=weights)[0]
        ok_prob = base_success[route]
        if degraded_tool_route and route == "tool_call":
            ok_prob = 0.70
        success = rng.random() < ok_prob
        if success:
            latency = max(80.0, rng.gauss(420, 120))
            error = None
        elif degraded_tool_route and route == "tool_call":
            latency = max(80.0, rng.gauss(1500, 300))
            error = "timeout"
        else:
            latency = max(80.0, rng.gauss(500, 150))
            error = "tool_error" if route == "tool_call" else "validation"
        traces.append({
            "trace_id": f"{release}-{i}",
            "release": release,
            "route": route,
            "success": success,
            "latency_ms": round(latency, 1),
            "error": error,
        })
    return traces


def main():
    rng = random.Random(7)
    baseline = generate_traces("v1.2", 1000, rng)
    candidate = generate_traces("v1.3", 1000, rng, degraded_tool_route=True)

    full = compare_releases(baseline, candidate)
    print(f"Full traffic (n={full['n_baseline']} vs {full['n_candidate']}): {full['verdict']}")
    print(f"  success: {full['success_baseline']:.3f} -> {full['success_candidate']:.3f} "
          f"(p={full['success_p_value']:.4f})")
    print(f"  latency p95: {full['latency_baseline']['p95']:.0f} -> {full['latency_candidate']['p95']:.0f}ms")
    for reason in full["reasons"]:
        print(f"  - {reason}")

    # The sampled-judge view: 5% of each release, as a judge would see it.
    sample_rng = random.Random(11)
    sampled = compare_releases(
        sample_rng.sample(baseline, 50), sample_rng.sample(candidate, 50)
    )
    print(f"\n5% sample (n=50 vs 50): {sampled['verdict']}")
    if sampled["verdict"] != "INCONCLUSIVE":
        print(f"  success: {sampled['success_baseline']:.3f} -> {sampled['success_candidate']:.3f} "
              f"(p={sampled['success_p_value']:.4f})")
    for reason in sampled["reasons"]:
        print(f"  - {reason}")

    # Route breakdown: where the regression actually lives.
    for route in ("chat", "tool_call", "retrieval"):
        b = [t for t in baseline if t["route"] == route]
        c = [t for t in candidate if t["route"] == route]
        print(f"  {route}: success {success_rate(b):.3f} -> {success_rate(c):.3f} "
              f"(n={len(b)} vs {len(c)})")


if __name__ == "__main__":
    main()
