# Adaptive reasoning effort: buy samples only when the vote disagrees

**Trending context.** Thinking Machines Lab released **Inkling 2 Small**
on 5 November 2026: a 594B-parameter sparse Mixture-of-Experts model
with only 17B parameters active per token, whose model card highlights
**variable reasoning effort** — the system can spend more or less
reasoning compute depending on the request (sources: MarkTechPost and
Unite.AI launch coverage, 5–6 Nov 2026; Inkling's benchmark figures are
theirs). Hosted on Tinker at launch, with Apache 2.0 weights promised
after external safety testing.

This demo isolates that one idea — variable reasoning effort — in pure
Python, with no model weights at all, so the budget-allocation logic can
be measured exactly.

## The setup

- A **problem** is a multi-step arithmetic chain (2–7 steps) with an
  exactly known answer.
- A **reasoning sample** walks the chain, but each step slips with
  probability 0.07 and perturbs the running value, the way a reasoning
  trace can go wrong mid-chain. A slipped trace lands on a wrong, and
  usually distinct, answer.
- **Self-consistency voting** takes the majority answer across samples;
  the winner's share of the vote is a cheap confidence signal.
- The **effort controller** buys samples in stages (3, then 5, then 9)
  and stops as soon as the vote reaches 0.80 agreement. Disagreement —
  not a difficulty guess — is what escalates the spend.

## Results (300 seeded problems, this code, run 6 Nov 2026)

| strategy | accuracy | avg samples | avg tokens |
|---|---|---|---|
| fixed_low (1 sample, no vote) | 0.770 | 1.00 | 52.7 |
| fixed_high (9 samples, always) | 0.997 | 9.00 | 474.5 |
| **adaptive (3 → 5 → 9, vote-gated)** | **0.997** | **5.75** | **318.5** |

Adaptive matched the always-max strategy exactly on accuracy while
using **36% fewer samples**. The single-sample baseline lost 23 points,
which is the cost of never checking your own work. Adaptive accuracy
stayed at 1.000 for chains up to 6 steps and only dipped at 7 steps
(0.977), where longer chains give slips more chances to compound.

## What this does and does not show

It shows the allocation principle behind variable reasoning effort:
agreement among independent attempts is a usable stopping signal, so
easy questions stop paying for compute they do not need. It does not
show how a real model calibrates effort internally — here the "model"
is a deliberately noisy calculator, and tokens are nominal (12 per
step). The transferable part is the controller and the measurement
harness, not the arithmetic.

## Run it

```bash
python3 adaptive_reasoning.py     # benchmark table
python3 -m pytest -q              # 10 tests
```

Pure standard library. Files: `adaptive_reasoning.py` (problem
generator, noisy sampler, vote, effort controller, paired benchmark),
`test_adaptive_reasoning.py` (chain correctness, no-slip rigged RNG,
vote confidence, ladder bounds, early stopping, determinism, and the
headline strategy comparisons).
