# Masked Diffusion Language Model (toy, numpy-only)

A from-scratch demonstration of the idea behind **Inception's Mercury 3**,
released 19 Oct 2026: a *diffusion* language model that generates text by
refining all positions **in parallel**, instead of writing one token at a
time left-to-right like every autoregressive model.

## The trending bit (all figures below are Inception's own claims)

Inception announced Mercury 3 on 19 Oct 2026 — a diffusion LLM the company
says matches GPT-6 Luna and Claude Haiku 5.5 on benchmarks (its own table:
GPQA Diamond 89.23, IFBench 76.84, SWE-Bench Atlas 51.08) while generating
at **1,068 tokens/sec** on NVIDIA GPUs, with a **1M-token context window**,
priced at $0.10/$0.35 per 1M input/output tokens at launch (50% off list).
Co-founder Volodymyr Kuleshov's launch post argues diffusion *scales*:
Mercury 3 is 40x the parameters of Mercury 1 with throughput "essentially
unchanged". Independent verification is still pending — treat the quality
numbers as the vendor's until third parties replicate them. What is not in
dispute is the mechanism, and that is what this repo demonstrates.

**Why diffusion is fast:** an autoregressive model needs one forward pass
per token (L passes for L tokens). A diffusion model starts from an
all-`[MASK]` draft and denoises every position at once, so it needs one
forward pass per *refinement step* — S passes with S << L. As answers get
longer, the gap grows.

## What this demo does

`diffusion_lm.py` implements absorbing-state (mask) discrete diffusion on a
tiny grammar (`DET NOUN VERB PREP DET NOUN` x2, e.g.
"the cat sat on the mat a dog ran near a tray"):

- **Forward process** `q(x_t | x_0)`: each token is independently replaced
  by `[MASK]` with probability `t` (the noise level).
- **Denoiser** `p(x_0 | x_t, t)`: token + positional + noise-level
  embeddings → single-head self-attention over observed tokens → residual
  MLP → per-position logits, trained with cross-entropy on masked positions
  only. Hand-written forward/backward passes (finite-difference
  gradient-checked, including the attention weights), Adam optimiser —
  no frameworks. Positions are added *before* attention so each slot's
  query knows where it is (a position-blind query made every masked slot
  attend identically and the model could not beat marginal guessing —
  found by test, fixed in this repo's history).
- **Reverse process** (confidence-based decoding, LLaDA/Mercury-style):
  start from all masks; each step, unmask the most confident predictions;
  repeat for S steps.

`sample.py` trains the denoiser and shows:

1. **Denoising accuracy vs noise level** — how much the model recovers at
   t = 0.2 / 0.5 / 0.8.
2. **Parallel generation** — an all-`[MASK]` draft becomes a full sentence
   in 4 refinement steps, printing the intermediate drafts.
3. **Infill** — fill the middle of a sentence while *both* ends are fixed
   (`the cat [MASK] ... [MASK] the mat`). This is the thing diffusion does
   naturally and left-to-right models cannot: the prediction sees the right
   side too.
4. **Cost scaling** — forward passes grow with steps, not length
   (e.g. ~8 passes for 240 tokens vs 240 for autoregression).

## Results (from `sample.py`, numpy on CPU)

| noise level t | masked-token recovery (400 held-out) |
|---|---|
| 0.2 | 0.563 |
| 0.5 | 0.502 |
| 0.8 | 0.436 |

Context: ~0.39 is marginal guessing on this grammar (most tokens are
independent draws) and ~0.51 is the ceiling even with perfect context —
the denoiser beats both at every noise level. The targeted
cross-position test is sharper: mask only the verb slots, keep subjects
visible. The verb is a deterministic function of the subject
(bird→flew, fish→swam, cat→sat, dog→ran), so a position-blind denoiser
scores 0.25 here; this one scores **1.000** — it genuinely reads context
through the attention layer.

Parallel generation: 12 tokens in 4 forward passes (vs 12 for
autoregressive), with intermediate drafts printed each step.

**The honest wrinkle — speed vs coordination.** Correlated slots are
predicted independently, so with only 4 refinement steps a verb can be
unmasked *before its subject resolves* and disagree with it
(`the bird sat near the mat`). More steps let every commitment see more
resolved context:

| refinement steps | sentences with subject-verb agreement (20 sampled) |
|---|---|
| 4 | 0/20 |
| 8 | 20/20 |

That knob — fewer steps for latency, more steps for coherence — is the
same tradeoff production diffusion LMs tune per use case. (A cheap
confidence-based corrective pass does *not* fix it: the bad verb was
committed confidently, so it is never selected for remasking. Confidently
wrong is the hard case.)

## Run it

```bash
python3 sample.py              # train + all demos (a minute or two on CPU)
python3 test_diffusion_lm.py   # 10 tests: masking stats, gradient check,
                               # training convergence, decode guarantees
```

Files: `diffusion_lm.py` (model + diffusion processes), `sample.py`
(demos), `test_diffusion_lm.py` (tests).

## Honest caveats

- Toy grammar, tiny model — this demonstrates the *mechanism*, not the
  quality/speed claims. Nothing here says anything about Mercury 3's
  benchmark scores; those are Inception's numbers, unverified.
- Real diffusion LMs (LLaDA, MDLM, Mercury) use transformers, noise
  schedules, and remasking strategies far beyond this sketch; confidence
  decoding here is the simple top-k variant.
- The "fewer forward passes" comparison counts model calls, not FLOPs —
  each diffusion pass processes the whole sequence.
