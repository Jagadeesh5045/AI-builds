# Looped transformer blocks — depth you rent, not buy

**Trending:** Thinking Machines Lab released **Inkling 2 Mini** on
20 November 2026 — an open-weight model built for local devices.
Reported figures (vendor / coverage, not measured here): a *looped*
Mixture-of-Experts transformer with 26.4B total and ~3.4B active
parameters, Apache 2.0 weights, a ~20 GB footprint that runs on a
consumer laptop, and the architectural twist this demo isolates —
**24 unique layers whose middle 12 are applied three times per token,
so one token sees 48 layer applications from 24 layers of weights.**

Coverage: Futurum Group (21 Nov 2026), Unite.AI / TestingCatalog via
Creati.ai (21 Nov), and the Navsplace inference notes (21 Nov), which
describe the 24-layer / middle-12-looped-3× structure. Where reports
differ in detail, this README treats all of them as reported claims.

## The idea

A normal transformer buys depth with parameters: 48 layers of depth
costs 48 layers of weights. A looped transformer *rents* depth with
compute instead: store a block once, apply it several times. The
trade-offs are exact opposites of a dense stack:

- **Memory** scales with the *unique* layers.
- **Latency / compute** scales with the *applications*.
- **Depth becomes a runtime knob** — the same weights can run more
  loops on a hard input and fewer on an easy one.

## What the code shows (`looped_transformer.py`, stdlib only)

The shared block is a max-propagation (BFS-style) operator over a
15-position line with the source re-injected every loop — the simplest
operator with the looped transformer's key property: **information
travels exactly one hop per application, so the loops you need are set
by the distance to the answer, not by the parameter count.**

Measured on the demo (seeded, deterministic):

| Loops | Positions reached | Far-end value |
|------:|------------------:|--------------:|
| 1 | 1 / 15 | 0.000 |
| 2 | 2 / 15 | 0.000 |
| 4 | 4 / 15 | 0.000 |
| 7 | 7 / 15 | 0.000 |
| 15 | 15 / 15 | 1.000 |

(The first application is the injection itself, so reach after *k*
loops is *k* positions, and the far end of a 15-cell line needs the
full 15 loops.)

**Adaptive halting.** With an ACT-style rule — halt a position once it
has received signal *and* stopped changing — the same 15-loop budget
produces the identical final state with **134 instead of 225
position-applications (40.4% saved)**. Loops used form a staircase
(2, 3, 4, …, 15, 15): positions near the source finish almost
immediately, only the far end spends the whole budget. That is the
looped model's version of adaptive compute.

**Parameter accounting.** For the reported Inkling 2 Mini shape
(24 unique, middle 12 ×3), applications per token = 6 + 36 + 6 = 48,
so an unlooped stack of the same effective depth stores **2.00×** the
layer parameters. Looping does not make compute cheaper — it makes
*memory* cheaper, which is exactly the constraint on a laptop.

## Two bugs the tests caught while building this

1. **An averaging block decays the signal by half per hop.** The first
   version mixed neighbours with a mean; after 14 hops the far end sat
   at ~0.00006 and "reach" stalled at 8/15. Switched to max-propagation.
2. **Halting on "no change" freezes unreached positions at zero.**
   A distant position does not change *because nothing has arrived
   yet*. The halt rule now requires received signal first — the same
   reason real halting mechanisms key on a learned halting score, not
   on raw state deltas.

## Honest limits

This is a propagation toy, not a transformer: no attention, no MoE
routing, no learned weights, and the "block" is hand-designed so its
per-loop behaviour is exactly analysable. Real looped transformers
learn what each loop should do, and whether later loops refine or
merely repeat is an empirical question this demo cannot answer. The
Inkling 2 Mini figures above are the vendor's and coverage's, labelled
as such; the only numbers measured here are the table, the halting
saving and the shape arithmetic.

## Run it

```bash
python3 looped_transformer.py        # demo
python3 test_looped_transformer.py   # 14/14 tests
```
