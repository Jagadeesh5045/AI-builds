# CoT Trace Guard: stopping adversarial distillation through reasoning traces

**Trending, 22 Nov 2026:** OpenAI is reported to have cut adversarial
distillation success rates from 54% to 0.2% by protecting intermediate
reasoning steps (chain-of-thought traces) in its latest models, as covered
in The Token Reporter's 22 Nov 2026 roundup. The threat model is simple:
if a model narrates *how* it decides, an attacker does not need the
weights. The narration is a labelled training set for a clone, and often
the decision procedure itself, spelled out.

This is a small, stdlib-only measurement of that leak, and of the cheapest
layered defence against it.

## The setup

A toy triage model holds a private rubric: a weighted score over four
features (`urgency`, `impact`, `sentiment_risk`, `account_value`) with a
threshold. Two servers expose it:

- **VulnerableServer** returns the label plus a full reasoning trace that
  restates the rubric weights and every per-feature contribution, the way
  a verbose chain-of-thought response restates its instructions.
- **GuardedServer** returns the label plus a number-free template
  explanation. Its draft text still passes through a `TraceGuard` scan
  (digit sequences and rubric vocabulary are blocked and counted), every
  response carries a canary token so leaked text can be recognised later,
  and a simple probe detector flags grid-sweep querying, the signature of
  label-based extraction.

Two attackers try to clone the model:

- **TraceAttacker** parses the weights and threshold out of a *single*
  vulnerable trace with regexes. No training, no query budget.
- **LabelOnlyAttacker** gets no traces, only labels, and fits a
  perceptron on 60 queried cases.

## Measured results (seeded, reproducible)

| Attacker | Access | Agreement with the true model |
|---|---|---|
| TraceAttacker | 1 reasoning trace | **1.000** (exact rubric recovered) |
| LabelOnlyAttacker | 60 labels | 0.927 (approximation, errors near the boundary) |

The guard recorded 0 redactions during the demo because the guarded
template is clean by construction; the scan is defence in depth for the
day a template or a model draft starts leaking numbers, and the unit
tests exercise it directly against leaky text.

## Honest limits

- A real rubric is not sitting in one trace as `0.42*urgency`; real
  extraction aggregates partial information across many traces. The toy
  compresses that into one response to make the mechanism visible.
- Label-only extraction is slowed, not stopped: with a much larger query
  budget the perceptron would converge further. That is why the probe
  detector and rate limits matter in production; trace hygiene removes
  the *cheap* path, it does not repeal model extraction.
- The reported 54% to 0.2% figures are OpenAI's, as reported; this demo
  measures its own toy numbers only.

## Run it

```bash
python3 trace_guard.py        # the comparison table
python3 test_trace_guard.py   # 12/12 tests
```

## Why this matters for LLM/RAG engineering

Any system that returns reasoning, retrieved-context scores, or reranker
weights to untrusted callers is publishing part of its decision
procedure. Audit what your API returns the way you would audit a training
data release: final answers out, intermediate scores and traces in.
