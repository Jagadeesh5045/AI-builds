# Attention Latent Feedback (ALF) in miniature

**Trending:** Xiong et al., *Attention Latent Feedback* (arXiv:2611.31154),
surfaced 24 Nov 2026 with code released. During autoregressive decoding, the
deeper layers have already computed rich representations of the *previous*
token, yet in a standard transformer the shallow layers at the current
position cannot read them: self-attention only connects positions at the
same depth. ALF gives every sublayer a learned, content-dependent read over
the deeper outputs of the previous position, blended into the usual layer
input by per-layer gates.

**What the paper reports** (its claims, on MoE models up to 4.0B parameters):

- held-out loss lower by 0.022 to 0.024 nats vs the standard residual,
- about 1.1% extra decoding FLOPs,
- on MoE-4.0B, average downstream accuracy 45.1% to 47.0%,
- on MoE-1.6B, matching a 1.38x larger baseline token budget.

## What this demo does

A NumPy miniature that isolates the mechanism, with no transformer:

- Each token gets a **shallow** feature (a linear map through a rank-4
  bottleneck, standing in for what an early layer has computed) and a
  **deep** feature (a two-layer tanh MLP, standing in for later layers).
- The label at position t is built from an interaction between the current
  token's shallow feature and the **previous token's deep feature**. That
  is precisely the information a same-depth architecture cannot reach
  early, and a fair miniature of the gap ALF targets.
- Both models use the same ridge head over `[current, previous,
  current x previous interaction]` features. The only difference is what
  the previous-position feature may be: shallow (baseline) or deep (ALF),
  with ALF's deep feature scaled per sublayer slice by learned gates.

## Results (seeded, run `python3 latent_feedback.py`)

```text
baseline (same-depth only)   test MSE: 0.0145   R^2: 0.748
ALF, gates fixed at 1        test MSE: 0.0025   R^2: 0.957
ALF, learned gates           test MSE: 0.0025   R^2: 0.957
relative MSE reduction (gated ALF vs baseline): 83.0%
estimated decoding overhead of the feedback read: 1.4% of layer FLOPs
```

ALF lands on the label's noise floor (the label noise variance is 0.0025);
the baseline cannot, however it weights the shallow previous feature.

**One honest caveat the demo exposes:** the learned gates move only
modestly from their 0.5 start. Because the ridge head is refitted after
each gate step, the head's weights absorb most of the gate scale, so gate
*magnitude* is near-redundant in this miniature. In the real architecture
the gates matter because the surrounding weights are shared and trained
jointly, not refitted per gate. The miniature's claim is narrower and it
holds: **access** to the deeper previous representation is what removes
the error, and it costs a gated vector blend, not a second attention pass
(estimated 1.4% of layer FLOPs here, against the paper's ~1.1% at scale).

## Run

```bash
python3 latent_feedback.py
python3 -m pytest test_latent_feedback.py   # or run the test functions directly
```

6 tests: shapes, determinism, ALF beating the baseline by a wide margin,
gates staying in range and helping, the feedback overhead staying in low
single digits, and a zero-gate control recovering baseline-level error
(proving the gain comes from the feedback content, not extra parameters).

## Why it matters for LLM/RAG engineering

Depth is usually treated as private to a position: layer 20's work on
token t is invisible to layer 3 at token t+1. ALF turns that finished work
into a cheap memory the next token can read. For agent and RAG workloads,
where consecutive tokens and steps repeatedly re-derive context, that
recycling is the interesting direction: more of the compute you already
paid for stays in play, at roughly one percent extra decoding cost.
