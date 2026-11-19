# Next-Concept Modeling: coarse-to-fine parallel generation, in miniature

**Trending:** Hazy Research (Stanford) published *Next-Concept Modeling:
Fast, Parallel Coarse-to-Fine Text Generation* on **19 Nov 2026**.
NCM learns to represent text as a hierarchy of discrete codes, then
generates **one level at a time** — predicting every code within a
level **in parallel** — instead of one token at a time left-to-right.
The paper reports competitive-or-better quality than strong diffusion
baselines while generating **10–31× faster** and seeing **4–5× fewer
training tokens** (paper figures, at model scale — not measured here).

## What this code shows

`next_concept.py` (stdlib only) builds the smallest honest version of
the schedule:

- A synthetic corpus of 240 sentences, each made of clauses from four
  topics (cooking, finance, health, weather) with varied wording.
- A two-level hierarchy: **level 1** is one concept (topic) code per
  clause; **level 2** is the exact detail tokens inside each clause.
  The decoder refuses a concept that disagrees with its details.
- Two generation schedules over the same codes: autoregressive
  (one token per sequential step) vs NCM-style (all concepts in one
  parallel step, then all details in one parallel step).

## Results (measured here, toy scale)

| Measure | Result |
|---|---|
| Sequential steps / sentence, autoregressive | 15.4 |
| Sequential steps / sentence, NCM (2 levels) | 2.0 |
| Sequential-step speedup | **7.71×** |
| Widest parallel step | 13.4 codes emitted at once |
| Exact reconstruction via concepts + details | 240 / 240 |
| Inside a 2-step budget | AR completes 480/3,700 tokens; NCM completes 3,940/3,940 codes |
| Distinct concept sequences in the corpus | 58 |
| Sentences fully pinned by concepts alone | 24.2% |
| Next-concept bigram accuracy (chance = 0.25) | 0.375 |

## Why it matters — and the caveats

- The win is a **schedule** win: sequential depth collapses from
  *tokens* to *levels*. That is the same intuition behind diffusion
  and speculative decoding, pushed into the representation itself.
- Coarse-to-fine only works if the coarse level is **learnable** and
  the details are **conditionally easy** given it. The bigram check
  (0.375 vs 0.25 chance) and the 24.2% concept-pinned share are toy
  probes of exactly those preconditions.
- Nothing here trains a network or reproduces the paper's quality
  results; it demonstrates where the parallelism comes from, what the
  extra concept codes cost, and why a wrong concept is expensive —
  the decoder rejects it rather than papering over it.

## Run it

```bash
python3 next_concept.py        # demo with the numbers above
python3 test_next_concept.py   # 12 tests
```
