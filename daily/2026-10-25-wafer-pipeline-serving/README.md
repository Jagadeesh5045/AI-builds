# Wafer-pipeline serving: reproducing Cerebras's 59-wafer arithmetic

Daily experiment, 25 October 2026. Standard library only, no model weights,
no API keys, nothing to download.

## The trending item

On 22 October 2026 Cerebras published a technical post on serving
trillion-parameter models on its WSE-3T wafers, using Moonshot's Kimi K3
(~3 trillion parameters, 93 layers) as its worked example: 44 GB of on-chip
SRAM per wafer, 4-bit weight quantisation, pipeline parallelism, and only
activations crossing between wafers (RDMA over Ethernet, ~2 microseconds
per hop), at a claimed 1,000-2,000+ tokens per second. Coverage on 25 October
2026 (ai0.news, LavX, Daily Inference) repeated those figures.

Two honesty notes, stated up front because they shape what this code is:

- Cerebras itself presents K3 as an illustrative example, and independent
  coverage notes the post does not supply a measured K3 run, benchmark
  conditions, or power figure for that layout. This demo reproduces the
  recipe's arithmetic; it does not verify the throughput claim.
- The 1.18 MW / ~4.1 joules-per-token figure is a Hacker News commenter's
  arithmetic, not a Cerebras measurement. The code below treats it as an
  input to check, not as a fact, and the check is the most interesting
  result here.

## What the code shows

`wafer_serving.py` (run it: `python3 wafer_serving.py`):

- **Weight budget.** 3T parameters at 4 bits = 1,500 GB total, 16.13 GB
  per layer over 93 layers, matching the post's "about 16 GB".
- **Wafer packing.** Dense (full-attention) layers take one wafer each so
  the rest of that wafer's SRAM holds the KV cache; linear-attention layers
  keep only a small fixed-size state, so two fit per wafer. 69 linear
  layers -> 35 wafers (the last holding a single layer) + 24 dense wafers
  = **59 wafers**. Each dense wafer leaves 27.87 GB spare for KV cache.
- **Pipeline bubble.** With 59 stages, 64 micro-batches give only 52.5%
  pipeline efficiency (47.5% bubble); 256 micro-batches raise it to
  81.5%. The bubble, not the hop latency, is the cost: all 58 hops add
  just 116 microseconds per full pass.
- **Quantisation cost.** On a seeded 64x64 Gaussian weight matrix,
  symmetric uniform quantisation gives 0.16235 relative Frobenius error
  at 4-bit, 0.00880 at 8-bit, and 0.00003 at 16-bit. The 4-bit saving that
  makes the model fit is real, and so is the error it buys.
- **Energy reconciliation.** 1.18 MW at 4.1 J/token implies 287,805
  tokens/s in aggregate, i.e. about 192 concurrent streams at 1,500
  tok/s (144 at 2,000 tok/s). A single stream at 1,500 tok/s on a
  1.18 MW rig would be ~787 J/token. So the commenter's J/token figure
  only makes sense as an aggregate across many concurrent requests,
  which is exactly what a pipelined wafer cluster is built to serve.

## Tests

`python3 -m unittest test_wafer_serving -v` — 17/17 passing, covering the
weight budget, the 59-wafer packing, pipeline efficiency, hop latency,
quantisation error ordering, KV-capacity scaling, and the energy
reconciliation.

## What this is not

Not a serving system, not a benchmark of Kimi K3, and not a claim that
1,000-2,000 tok/s was measured. The KV-capacity helper takes
bytes-per-token as an input because the post does not state Kimi's hidden
size; any token count from it is only as good as that assumption.
