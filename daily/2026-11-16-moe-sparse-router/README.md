# Sparse MoE Router — Top-k Experts, Load Balancing and Capacity

A pure-Python demonstration of sparse Mixture-of-Experts routing: why
a model can hold trillions of parameters yet only pay for a small
slice of them on each token.

## Why this, why now

On 16 November 2026 Alibaba released Qwen4-Max, its new Qwen
flagship. Launch coverage and Alibaba Cloud documentation describe
it as a Mixture-of-Experts model with 2.4 trillion total parameters
activating about 71 billion per token, a 1,000,000-token context
window, and multimodal (text, image and video) input. Unlike earlier
Qwen flagships it is closed-weight and API-only; Alibaba says
smaller Qwen4 models will be released open-weight later, with no
date given.

On cost, Alibaba reports its own Work360 test falling from $4.43 to
$1.67 average cost per task against Qwen3.8-Max. All figures in this
paragraph are Alibaba's own claims, not independently verified
results, and benchmark scores depend on the tests chosen.

## What the code shows

`moe_sparse_router.py` (standard library only) implements:

- **Top-k routing** — a linear router scores all 8 experts per
  token; only the top 2 run. Compute tracks active parameters, not
  total parameters.
- **Load-balancing loss** — the Switch-style auxiliary loss
  `E * sum(mean_prob_i * fraction_i)`, which is 1.0 for a perfectly
  balanced router and rises as routing collapses onto a few experts.
  Without it, routers collapse in exactly that way.
- **Expert capacity** — each expert takes at most
  `ceil(capacity_factor * tokens * top_k / experts)` tokens.
  Overflow routing slots drop to the residual path and are counted,
  so a hot expert can never silently double-process work.
- **Active-vs-total accounting** — active fraction, a forward-pass
  FLOPs proxy (about 2 FLOPs per active parameter) and a
  cost-per-task proxy scaled by active size against a priced
  reference model. Planning proxies, not billing formulas.

## Measured demo run (seeded, reproducible)

512 synthetic tokens, 8 experts, top-2 routing, capacity factor 1.25:

- Expert load: `[154, 160, 63, 134, 0, 62, 137, 160]`
- Load imbalance (max / mean): **1.47** — and expert 4 received no
  tokens at all, which is the collapse the balancing loss exists to
  punish
- Dropped routing slots at capacity: **154** of 1,024
- Auxiliary balancing loss: **1.144** (1.000 = perfectly balanced)
- Demo layer active share: **26.2%** of parameters per token
  (2 of 8 experts plus the router)
- Qwen4-Max, on Alibaba's reported figures: 71B active of 2,400B
  total = **3.0%** per token

The gap between the demo's 26.2% and the reported 3.0% is the
point: production MoE uses far more, far larger experts with a
small top-k, so the active slice shrinks while total capacity —
knowledge stored across all experts — keeps growing.

## Honest caveats

- The demo experts are tiny seeded MLPs on synthetic clustered
  tokens. Nothing here trains a model, measures real quality, or
  reproduces Qwen4-Max in any way.
- The FLOPs and cost figures are arithmetic proxies. Real serving
  cost also depends on memory bandwidth, expert parallelism,
  communication between devices, batching and the price actually
  charged per token.
- A dropped token in this demo falls back to the residual path;
  production systems vary (some reroute, some drop differently) and
  dropping does cost quality on the affected tokens.

## Run it

```bash
python moe_sparse_router.py
python -m pytest -q   # 16 tests
```
