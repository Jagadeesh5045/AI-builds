# experiment-lab

An A/B testing and experimentation statistics toolkit, demonstrated end-to-end on a
realistic e-commerce checkout experiment. Built as a Data Scientist weekly project:
the statistics, experimentation rigour, and analytics that decide whether a product
change actually worked.

## Problem statement

Most A/B test writeups stop at "p < 0.05, ship it". Real experimentation is harder:
dashboards get peeked daily (inflating false positives), underpowered tests miss real
effects, high-variance metrics need huge samples, and a single p-value hides the
business question. experiment-lab implements the full honest toolkit and applies it
to one experiment, so every claim can be checked.

## Real-world use case

A simulated checkout-redesign experiment: 4,000 users randomised 50/50 over 14 days,
with daily seasonality, weekend traffic dips, a pre-experiment spend covariate, a
binary conversion outcome and revenue per user. The question: did the redesign lift
conversion, and should we ship it?

## Approach

1. **Guardrails first** — sample-ratio-mismatch chi-square check before any outcome is
   inspected, so broken randomisation can't masquerade as a result.
2. **Primary metric** — two-proportion z-test on conversion rate with a 95% confidence
   interval for the lift.
3. **Peeking audit** — re-runs the test on cumulative data after each day, showing
   which days a naive dashboard-peeker would have declared victory early.
4. **CUPED** — regresses the metric on pre-experiment spend to cut variance, applied
   to revenue per user and to average order value (converters only).
5. **Bayesian read** — beta-binomial posteriors: P(treatment better), expected lift,
   95% credible interval.
6. **SPRT** — sequential probability ratio test: the honest way to stop early, with
   explicit accept/reject boundaries.
7. **Power context** — minimum detectable effect at this sample size, and the sample
   needed for 80% power at the observed lift.

## How to run

```bash
pip install -r requirements.txt
python run_analysis.py        # generates data, runs analysis, writes report.md + figures/
pytest tests/ -q              # 13 unit tests for the statistics
```

## Results (actual output, seed 42, fully reproducible)

- Control: 2,006 users, 5.93% conversion. Treatment: 1,994 users, 7.77%.
- **Lift: +1.84 percentage points (+31.0% relative), 95% CI [+0.28, +3.41], z=2.30, p=0.0212. Verdict: ship it.**
- Randomisation healthy: SRM chi-square p=0.850.
- Peeking audit: a naive dashboard would have called it on days 11-14, still before the
  fixed horizon, but the point stands, only the pre-registered test counts.
- Bayesian: P(treatment better) = 98.9%, expected lift +1.84pp, 95% CrI [+0.27, +3.42].
- CUPED on average order value: 34.9%/34.5% variance reduction, CI width GBP 64.18 down
  to GBP 51.86. Adjusted AOV lift GBP -9.00 (p=0.50): the redesign lifts conversion,
  not basket size. Honest nulls get reported too.
- SPRT stopped early with a significant-lift decision after 670 of 1,994 pairs.
- Power context: MDE 2.24pp at this size; 2,955 users per group needed for 80% power
  at the observed lift.

## Tech stack

Python, pandas, NumPy, SciPy, matplotlib, pytest.

## Project structure

```
experiment-lab/
├── experiment_lab/
│   ├── generate.py     # seeded synthetic checkout-experiment dataset
│   ├── stats.py        # z-test, Welch t, SRM, power/MDE, Bayesian A/B, CUPED, SPRT
│   ├── analysis.py     # full pipeline + markdown report writer
│   └── visualize.py    # daily trends, posteriors, CUPED comparison figures
├── tests/test_stats.py # 13 unit tests (13 passing)
├── data/experiment_sample.csv  # 500-row sample (full 4,000-row set regenerates via seed 42)
├── figures/            # generated charts (see report.md)
├── run_analysis.py     # entry point: data -> analysis -> report + figures
└── report.md           # generated experiment report
```
