# Hybrid SSM + Attention: selective state spaces from scratch

**Trending this week:** NVIDIA released **Nemotron 3.5 Super VL** (120B open
multimodal model, hybrid **Mamba-2 + attention + MoE** architecture, C-RADIO
vision encoder, distilled with speculative decoding and NVFP4
quantization-aware training; 74B **Bolt** variant in native 2-bit so it fits a
24GB GPU, launched alongside RTX Spark notebooks on Oct 16), and Meta launched
**Muse Spark Open** (816B total / 42B active, 1M context, open weight). The
industry direction is clear: giant open-weight models, hybrid
state-space/attention stacks, and aggressive quantization so they run on
hardware people actually own.

This repo implements the two core ideas behind the "hybrid" half of that
story, from scratch in NumPy (no torch, no jax):

1. **`ssm.py`** — a selective state-space layer (Mamba-style S6): the
   recurrence step size Δ and the input/output projections B, C are *functions
   of the input*, so each token decides what the state remembers vs forgets.
   Zero-order-hold discretization, exact reverse-mode backprop (gradient
   checks in `test_ssm.py`).
2. **`attention.py`** — single-head causal self-attention baseline, exact
   backprop.
3. **`selective_copy.py`** — the canonical Mamba-paper task: noise tokens +
   sparse (marker → needle) pairs; queries ask "what followed marker m?".
4. **`train_compare.py`** — trains both on selective copy, then compares
   accuracy, a selectivity probe (does Δ spike on salient tokens?), and
   forward+backward wall-time scaling O(L) vs O(L²).
5. **`vram_fit.py`** — memory-fit calculator for the week's open-weight
   releases: weights + KV-cache arithmetic answering "can I run this locally?"

## Results (measured on this machine, NumPy, toy scale)

### 1. Content-addressed recall: one layer vs two

Task: 48 noise tokens + 3 (marker → needle) pairs at random positions, then
queries ask "what token followed marker m?" 800 Adam steps, batch 32, D=32,
one seed per model.

| model   | final loss | held-out acc |
|---------|------------|--------------|
| ssm-1L  | 2.03       | 0.193        |
| attn-1L | 2.05       | 0.184        |
| attn-2L | 0.52       | 0.846        |

Both single-layer models sit at the "uniform guess over the 8 needle tokens"
floor (ln 8 ≈ 2.08; chance accuracy 0.125). The two-layer attention model
escapes it with a sharp phase transition around step 500 — the classic
induction-head circuit forming: layer 1 learns to shift previous-token
information forward, layer 2 matches the marker and copies the needle.

Why the single layers fail is architectural, not a tuning issue: the query is
a fixed function of the marker token while the needle is random per sequence,
so no fixed query/key pair can systematically match the right position — the
marker→needle binding needs the two-step circuit. The SSM's selectivity
mechanism is faintly visible (mean Δ: 0.063 on markers vs 0.051 on noise) but
cannot create the cross-position binding either. This is the honest case for
hybrid stacks like Nemotron 3.5's: SSM layers for cheap O(L) long-range
filtering, attention layers for the precise associative retrieval that RAG
pipelines and agents depend on.

### 2. Scaling: O(L) vs O(L²)

Forward+backward wall time in ms (batch=2, D=64, N=32):

| L    | SSM  | attention | ratio |
|------|------|-----------|-------|
| 256  | 157  | 126       | 0.80x |
| 512  | 777  | 516       | 0.66x |
| 1024 | 2391 | 2959      | 1.24x |
| 2048 | 2099 | 10034     | 4.78x |

Attention's per-doubling growth (3.4–5.7x) tracks the quadratic cost of
materializing the L×L score matrix; the SSM folds each token into a
fixed-size state. Timings are noisy on a shared VM, but the 4.8x gap at
L=2048 is the whole argument for SSM layers in long-context stacks.

### 3. vram_fit.py: can you run this week's open weights locally?

- Nemotron 3.5 Bolt 74B-A9B @ 2-bit: 17.2 GB weights + ~2 GB KV cache at 8k
  context ≈ 19 GB → fits a 24 GB GPU (matches NVIDIA's claim).
- Muse Spark Open 816B @ 4-bit: 380 GB weights → needs the ~1 TB RAM setup
  TechCrunch described; no consumer GPU comes close.

## Run it

```bash
python test_ssm.py        # gradient checks + smoke training
python train_compare.py   # full experiment (~a few minutes)
python vram_fit.py        # memory-fit table for Nemotron 3.5 / Muse Spark Open
```

## What the numbers mean

- The recall experiment (§1) is the architectural argument for hybrid
  models, demonstrated rather than asserted: precise content-addressed
  retrieval needs the two-step induction circuit (shift-then-match-and-copy),
  which a single layer of either family cannot express. SSMs earn their place
  with linear-time long-range filtering (§2); attention earns its place with
  exact associative recall. Nemotron 3.5 ships both layer types for exactly
  this reason.
- The selectivity probe shows the Mamba-style gating is real but limited:
  Δ rises on salient tokens (markers 0.063 vs noise 0.051), i.e. the layer
  learns *what to let into the state*, but an additive fixed-size state still
  cannot bind "marker m goes with needle n" across positions.
- `vram_fit.py` is the practical companion: the same week's releases only
  make sense with quantization arithmetic — 2-bit is what puts a 74B model on
  a 24 GB card, and nothing puts 816B on one.

## Honest caveats

- Toy scale: D=32, N=16, L=51, single layer, synthetic task. This
  demonstrates the *mechanism*, not the real models.
- Simplified vs Mamba-2: no SSD (state-space duality) block algorithm, no
  head structure, simplified B̄ = ΔB discretization. The real Mamba-2 is a
  hardware-aware parallel scan; this is the math it computes.
- Vendor figures (Nemotron's throughput claims, Bolt's "fits 24GB", Spark's
  1TB-RAM requirement) are *their* claims, cited as such. The byte counts in
  `vram_fit.py` are my own back-of-envelope arithmetic (params × bits/8 plus
  a standard KV-cache formula with stated layer/head assumptions).
- Training curves are from one seed each; rerun with another seed and the
  exact numbers move.
