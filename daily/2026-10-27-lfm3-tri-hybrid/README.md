# LFM3 tri-hybrid sequence mixer (27 October 2026)

## The trending item

On 27 October 2026 Liquid AI released LFM3, a family of six open-weight
models in the 2B class (text, vision, audio, omni and two decision
variants) branched from one 1.7B base. Launch coverage reports the text
backbone as a tri-hybrid: 13 gated short-convolution layers, 5
sliding-window attention layers with a 256-token window, and 6 global
attention layers across 24 blocks, kernel size 7, a 256K context window,
pre-training on 22 trillion tokens, and a roughly 1 GB footprint at 4-bit
precision. Liquid AI says the 1.7B text model beats its earlier LFM2.5-2.6B
with 35% fewer parameters; those are the company's claims, not results
reproduced here.

Coverage:
- https://aiweekly.co/alerts/liquid-ai-opens-six-model-lfm3-family-at-256k-context
- https://dailyaibrief.com/news/liquid-ai-releases-lfm3-open-weight-on-device-models-cAGWxkSa

## What this demo is (and is not)

This is a small NumPy re-implementation of the tri-hybrid *pattern* only:
the 13 / 5 / 6 layer split, causal gated depthwise convolutions,
sliding-window attention and global attention, with residual connections
and per-token layer normalisation. It contains no Liquid AI code and no
LFM3 weights, and it runs at toy dimensions on CPU. The question it
answers is architectural: what does each layer type contribute, and why
is the mix cheaper than all-global attention at long context?

## What it shows

Three probes, all passing:

1. **Locality.** A single impulse through a gated convolution (kernel 7)
   affects exactly positions 3 to 9, the impulse position plus the next
   six positions, and nothing before or after.
2. **Reach.** With identity query/key weights, the last token puts
   weight 1.0000 on a strong first-token key under global attention, and
   exactly 0.0000 on it under a sliding window of 4, which cannot reach
   that far back.
3. **Causality.** Perturbing only the last token of a 32-token sequence
   changes earlier outputs of the full 24-layer stack by exactly 0.0,
   while the last position changes by 0.6285. Every layer type here is
   causal, so no future information leaks backwards.

Mixing-cost scaling (d_model 128, window 256, kernel 7, counting scores
plus weighted sum per layer):

| Sequence length | Full attention | Tri-hybrid | Hybrid share |
|---:|---:|---:|---:|
| 1,024 | 0.0 TFLOP | 0.0 TFLOP | 0.306 |
| 16,384 | 1.6 TFLOP | 0.4 TFLOP | 0.253 |
| 65,536 | 26.4 TFLOP | 6.6 TFLOP | 0.251 |
| 262,144 | 422.2 TFLOP | 105.6 TFLOP | 0.250 |

At 256K context the hybrid mix costs 25.0% of the all-global-attention
baseline, because only 6 of 24 layers pay the quadratic price; the
convolution and sliding-window layers grow linearly with sequence
length instead. The global layers are still quadratic, so this is a
fourfold saving on mixing cost, not a removal of the quadratic term.

## Run it

Requires NumPy only:

```bash
python demo.py            # probes + scaling table
python test_tri_hybrid.py # 8 tests, all passing
```

## Files

- `tri_hybrid.py` — gated depthwise convolution, window/global
  attention, the 24-block stack, and the FLOPs comparison.
- `demo.py` — the three probes and the scaling table.
- `test_tri_hybrid.py` — 8 unit tests (pattern counts, convolution
  locality, attention causality, window masking, stack shape and
  causality, FLOPs saving).
