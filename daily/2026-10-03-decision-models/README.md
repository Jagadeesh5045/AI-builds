# Decision models: probability over actions, not text

**Trending:** On 1 October 2026 Cloudflare launched **Clef and Clef-flash**,
open-source "AI decision models" (Apache 2.0, Qwen-based) built for autonomous
agents: they output a **probability for each possible action** instead of
generating text, making operational decisions in ~39 ms and deferring to a
human when unsure. The same week, llama.cpp added decision-model support via a
`/v1/systemone` endpoint. The pattern is shifting from "ask an LLM and parse
the answer" to "one forward pass, probabilities out".

This experiment implements that pattern end to end with a classical-ML
decision head (numpy only, no API keys needed):

`agent context vector -> DecisionHead -> P(action) distribution`

## What is in here

| File | What it does |
|---|---|
| `decision_head.py` | `DecisionHead`: multinomial logistic regression (numpy) that returns a probability per action, with **confidence gating** (below the threshold it routes to `escalate`, i.e. a human hand-off, mirroring Clef's defer-to-human behaviour), calibration diagnostics, and latency benchmarking |
| `data_gen.py` | Synthetic agent telemetry: 6 features (risk, context completeness, tool availability, budget, ambiguity, success history) and the "correct" action under operator-style rules |
| `demo.py` | Trains, shows calibration tables, runs 5 example scenarios (including a borderline case that gets gated to a human), and prints latency vs. a simulated LLM round trip |
| `test_decision_head.py` | Tests: softmax/probability sanity, loss reduction, reproducibility, gating behaviour, payload shape |

## Why this matters for agents

Text-generating LLMs are slow and their decisions need parsing (and parsing
breaks). A decision head is cheap, deterministic to evaluate, and its
probabilities are auditable: you can log exactly how confident the agent was
in each option. In production you put the head *in front of* the LLM: cheap
calls go straight through, low-confidence ones escalate, and the LLM handles
only what the head cannot.

## Run it

```bash
pip install numpy          # only dependency
python test_decision_head.py   # 6 tests
python demo.py
```

Sample output:

```
Trained 400 epochs | final loss 0.6698 | train accuracy 0.747
Holdout accuracy: 0.730
  Example decisions:
    routine safe task            -> proceed         conf=0.97
    vague brief, little context  -> gather_context  conf=0.99
    risky payment action         -> escalate        conf=0.91
    tools down, budget gone      -> defer           conf=0.88
    borderline: risky but well-understood -> escalate conf=0.54  [GATED TO HUMAN]
  Decision head: 0.0xxx ms per decision  -> thousands of x faster than an LLM round trip
```

## Trend links

- Cloudflare Clef / Clef-flash launch (1 Oct 2026): open decision models, 39 ms, probabilities over text
- llama.cpp decision-model support (`/v1/systemone`) in llama-server
- The broader shift: agents that run unattended need *controllable* decisions, and calibrated probabilities give you that control

## Ideas for next

- Fit a temperature or isotonic calibrator on the head's outputs to push ECE lower
- Swap the logistic head for a tiny MLP or gradient-boosted trees
- Serve it behind a `/v1/systemone`-style JSON endpoint (FastAPI)
