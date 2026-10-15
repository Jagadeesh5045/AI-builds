# Probabilistic Bias Correction (PBC) — toy demo

**Trending hook:** on 14 October 2026 a Microsoft Research-led team published a
*Nature* paper introducing *probabilistic bias correction* (PBC): an ML
framework that learns to correct a forecast model's *historical probabilistic
forecasts*, operating directly on probability distributions rather than raw
observations. The team — Hannah Guan and Soukayna Mouatadid, with Judah Cohen
(MIT) and Lester Mackey (Microsoft Research) as corresponding authors, plus
ECMWF and other collaborators — reports that PBC lifted accuracy on 98% of
precipitation targets and 92% of temperature targets for ECMWF's top
physics-based system, doubled the skill of ECMWF's own AI forecaster, and, for
the first time, beat climatological averages on the 2–6 week subseasonal
horizon. The authors released the code open source and note it runs on ordinary
CPUs. (Coverage, 14–15 Oct 2026:
[MIT News](https://news.mit.edu/2026/researchers-improve-weather-forecasts-looking-two-six-weeks-out-1014),
[Unite.AI](https://www.unite.ai/microsoft-research-publishes-ai-weather-forecasting-study-in-nature/).)

**This folder is an independent, stdlib-only re-implementation of the *idea* on
synthetic data — not the authors' code. Every number below comes from a toy
world, not the paper, and the paper's performance claims are the authors' as
reported.**

## The idea in one line

Given a model's past forecast CDFs F_1..F_n and the outcomes y_1..y_n, the
probability-integral-transform (PIT) values z_i = F_i(y_i) should look uniform
if the model is probabilistically calibrated. Systematic deviation of the PIT
distribution from uniform *is* the model's distributional bias. PBC learns the
empirical CDF G of past PIT values and corrects each new forecast as
`F_corrected(x) = G(F_raw(x))` — a correction applied directly to the
probability distribution, the paper's headline move.

## What the demo does

- `pbc.py` builds a toy world: daily temperature anomaly = seasonal cycle +
  AR(1) noise, plus a biased "dynamical" forecaster with systemic flaws that
  mirror real NWP biases (damped extremes, warm offset with slow drift,
  overconfident spread).
- `PBCorrector` learns G from a 3-year archive of (forecast, outcome) pairs —
  using Weibull plotting positions so the remap is continuous and monotone —
  then corrects each holdout forecast distribution via `G(F_raw(x))`.
- Scores raw vs PBC-corrected vs climatology on a 1-year holdout with CRPS
  (closed form for Gaussians, split-interval numerical integration for the
  corrected CDFs), Brier score for P(y > 5), and PIT-histogram uniformity.

## Results (seed 7 — deterministic)

| metric (lower is better) | raw model | PBC corrected | climatology |
|---|---|---|---|
| CRPS | 2.1947 | **1.7140** (−21.9%) | 3.5223 |
| Brier, P(y>5) | 0.1252 | **0.1119** | 0.2020 |
| PIT chi-square (10 bins) | 588.01 | **6.59** | — |

PBC cuts CRPS by 21.9% versus the raw model and beats climatology on the
holdout — the same "beats climatology" headline the paper claims for the real
2–6 week horizon.

## Run it

```bash
python3 pbc.py                  # prints the demo scoreboard
python3 -m unittest test_pbc   # 21 tests, all passing
```

## Files

- `pbc.py` — Gaussian helpers, toy world + biased forecaster, `PBCorrector`,
  proper scoring rules, demo scoreboard. No third-party dependencies.
- `test_pbc.py` — 21 deterministic tests: closed-form vs numerical CRPS,
  CDF validity/monotonicity of the remap, CRPS + Brier + PIT improvements on a
  held-out set, reproducibility, no train/holdout leakage by construction.

## Honest caveats

- The toy world is Gaussian and one-dimensional; real NWP errors are
  spatiotemporal and non-Gaussian, which is why the real paper's ML machinery
  is far richer than PIT remapping.
- PIT remapping fixes *marginal* calibration as seen in the archive; it cannot
  invent sharpness the base model never had, and it assumes the bias regime is
  stationary between archive and deployment.
