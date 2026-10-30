# Agent release-health monitor (no LLM judge)

**Trending hook:** on 29 October 2026, Oimo AI open-sourced Apgar, a
release-health checker for AI agents that watches production (Langfuse)
traces and compares releases on observable metrics, instead of paying
for an LLM judge over a sample of traffic. This is an independent,
from-scratch re-implementation of that idea in the Python standard
library. It is not Apgar's code, and it uses no real Langfuse data:
the traces in the demo are synthetic, generated with a fixed seed.

## What it does

`release_health.py` groups traces by the release recorded in their
metadata (automatic release detection, no manual tagging), then
compares a candidate release against a baseline on three surfaces:

1. **Success rate** — two-proportion z-test, so a drop must beat noise.
2. **Latency** — mean, p50, p95; a slow release can be a broken release.
3. **Error mix** — a new error type, or a sharp rise in an old one.

Verdicts: `REGRESSION`, `NO REGRESSION`, `WATCH` (moved, but not past
a threshold), or `INCONCLUSIVE` (fewer than 50 traces per side).
Thresholds: success fall >= 3pp at p < 0.05, p95 latency rise >= 20%
with mean rise >= 10%, or any error-type share rise >= 5pp.

## Run it

```bash
python3 release_health.py
python3 -m unittest test_release_health -v
```

## Demo results (seed 7, 1,000 synthetic traces per release)

The candidate release breaks one route: tool calls fall from 96% to
70% success in the generator, and their failures become slow timeouts.
Chat and retrieval are unchanged.

| Signal | Baseline v1.2 | Candidate v1.3 |
| --- | --- | --- |
| Success rate | 0.939 | 0.889 (5.0pp fall, p = 0.0001) |
| Latency mean / p95 | 422 / 626 ms | 501 / 1,331 ms |
| Timeout error share | 0.000 | 0.076 |
| **Verdict (full traffic)** | | **REGRESSION** (all three signals) |

Route breakdown: tool_call 0.952 -> 0.690, chat 0.946 -> 0.958,
retrieval 0.898 -> 0.941 (the last two are sampling noise, which is
exactly why the verdict needs a significance test, not raw deltas).

The 5% sampled view (n = 50 vs 50) still flags the release, but only
through the timeout error alone: its success-rate evidence is
inconclusive (0.940 -> 0.880, p = 0.2945), against p = 0.0001 on full
traffic. Sampling weakens the headline metric; full-traffic observable
metrics do not need a judge at all.

## Files

- `release_health.py` — grouping, statistics, comparison, demo.
- `test_release_health.py` — 13 unit tests (grouping, percentiles,
  z-test, each verdict path, small-sample handling).

## What I would do next

Point `group_traces` at a real trace export (Langfuse JSON, or any
`{release, success, latency_ms, error, route}` rows), add a
per-route breakdown to `compare_releases`, and alert on `REGRESSION`
before a canary is promoted, not after.
