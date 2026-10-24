# Kimi K3.1 Pro variant router + quality gate

A small, correct toy of the deployment question Moonshot AI's
23 October 2026 launch creates: Kimi K3.1 Pro ships as *two tiers with
the same capability* (Standard, and a Highspeed tier), so which tier
should a given task pay for, and did its output pass automated checks?

## What actually launched (claims labelled)

- **Moonshot's own words:** Kimi K3.1 Pro is "the first model in our
  new K3.1 series and our most capable 3T-class open-weight model to
  date, with a 1-million-token context window", targeting coding,
  3D creation, financial research and modelling, and long-horizon
  agent work. A faster sibling, Kimi K3.1 Pro Highspeed, shipped the
  same day (API model IDs reported as `kimi-k3.1-pro` and
  `kimi-k3.1-pro-highspeed`).
- **Reported, not Moonshot-confirmed:** The Decoder reports the model
  keeps K3's dimensions (2.8 trillion total parameters, 104 billion
  active), and that Highspeed runs 2-3x faster with the same
  capabilities. Pricing via The Decoder: Standard $2 input / $10
  output / $0.20 cache-hit per million tokens; Highspeed $4 / $20 /
  $0.20. Weights are *not* published yet; a 4 November release is a
  reported target, not confirmed in Moonshot's own materials.
- **Benchmarks (Moonshot's table, as reproduced by The Decoder,
  reasoning effort max):** Terminal-Bench 4.0 65.7, SWE Marathon
  51.7, BenchCAD with tools 92.7, OfficeQA Pro v2 64.4, ExploitBench
  74.7 (the cyber gap is deliberate: Moonshot says it removed
  cyber-related training data for the open-weight release).
  Artificial Analysis, cited by The Decoder, scores it 55 on its
  Intelligence Index, fourth overall and 11 points above Kimi K3.
- **The lesson this demo encodes:** despite a 33% per-token price cut
  versus K3, the reported Artificial Analysis task cost barely moved
  ($2.00 -> $1.98), because the new model generates ~14% more output
  tokens. Measure cost per *finished task*, not per token.

## What the code does

`router.py` (stdlib only):

- **Pricing + latency model** for the two tiers, with cached input
  billed at the same $0.20 rate in both (so the Highspeed premium
  shrinks on cache-heavy work).
- **`route(task)`** with explicit, priority-ordered rules: (1) a
  deadline Standard cannot meet, (2) interactivity (urgency >= 0.8),
  (3) cache-heavy work (>= 50% cached) at moderate urgency, (4)
  Standard by default for batch work.
- **`check_output(output)`**: a serving-time quality gate in the
  spirit of the automated post-training checks Moonshot describes.
  It fails outputs with a missing required field, a placeholder
  (`TODO`/`TBD`/etc.), or a numeric check outside tolerance.
- **`generation_cost_factor` / `compare_k3_k31`**: the arithmetic
  behind the lesson above. Output-cost multiplier after the price
  cut plus 14% token growth is (10/15) x 1.14 = **0.76**, not 0.67.

## Verified results (run locally, 24 Oct 2026)

- `python3 test_router.py`: **17/17 tests pass**.
- `python3 demo.py` on an 8-task mixed workload (coding, finance,
  CAD, agent work): **5 tasks routed to Highspeed, 3 to Standard**;
  routed total **$2.7870 in 21.94s**, versus $3.1970 all-Highspeed
  (**saves $0.4100, 12.8%**). Honest note: routing costs *more*
  than all-Standard ($1.6870) -- that is the point, it buys latency
  where a deadline or a waiting user justifies it, and nowhere else.
- Quality gate on 4 sample outputs: **1/4 passed, 3 rejected**
  (missing `total` field; `TODO` placeholder; revenue 118 vs 100).
- Price-vs-task-cost example (2k input, 10k output): K3 $0.1560 vs
  K3.1 Pro $0.1180 at 11,400 output tokens.

## Caveats

This is a toy: latency constants are invented, no real API is
called, no Kimi weights are used (they are not public yet), and
all vendor/third-party figures above are as reported on 24 Oct
2026 -- verify current pricing on the official Kimi API platform
before budgeting anything real. This demo is also deliberately
different from the 5 Oct MoE-routing experiment: that one routed
*tokens between experts inside* a model; this one routes *tasks
between deployment tiers* and gates their outputs.
