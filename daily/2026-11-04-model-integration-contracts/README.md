# Model Integration Contracts — a tiny harness that catches a silent attention regression

**Trending (3 Nov 2026):** Hugging Face proposed **Model Integration Contracts (MICs)** after a reduced-size UMT5 fixture detected a decoder attention regression in Transformers that ordinary forward-pass checks had missed.

## What Hugging Face reported (their figures, not mine)

- A change on **30 July 2026** enabling PyTorch scaled dot-product attention for T5 had a training-time side effect: the UMT5 decoder could **attend to future tokens**, including tokens it was supposed to predict. Nothing crashed.
- Replaying frozen test contracts over **five months of Transformers commits** flagged the regression from its introduction through its fix on **28 September 2026**.
- The fixture showed a **1.22 difference in label log-probabilities** from a reviewed baseline, and a training-loss difference of about **0.20**.
- The experimental fixture used **66 MB of weights vs 1,227 MB** for the source checkpoint — important in a library Hugging Face says supports **750+ model types**, with **88,000+ CPU tests** on the main branch and **200,000+ GPU tests** in a daily run. Running every full integration test on every pull request is not affordable; a small contract is.
- A proposed `transformers-mic` format packages a contract as three parts: a **reduced-size fixture** based on a real checkpoint's configuration, a **runnable example** showing how the model is used, and an **output baseline reviewed by a person**.

Caveat, stated plainly in the coverage: MIC is a *proposed* standard. Its broader detection rates, false-alarm rates, adoption and upkeep costs across the model catalogue are not yet established. This demo tests the core idea on one failure class, not the whole proposal.

## What this demo does

`mic_harness.py` (stdlib + NumPy, deterministic) builds a tiny decoder-only transformer twice with **identical weights**:

- `causal=True` — the correct model, triangular attention mask on.
- `causal=False` — the regression: the mask is dropped, so position *t* can see tokens after *t*. Same shapes, same code path, no crash — the exact failure class described above.

It then freezes a contract from the correct model (fixture config + one runnable teacher-forced example + a reviewed baseline, tolerance ±0.05) and runs three probes:

1. **Causality** — rewrite every token after a cut point; logits at or before the cut must not move (tolerance 1e-6).
2. **Baseline drift** — teacher-forced label log-prob mean vs the frozen baseline.
3. **Loss drift** — fixture-batch cross-entropy vs the frozen baseline.

Finally it **replays** the unchanged contract over a synthetic five-commit history (good, good, regressed, regressed, fixed), mirroring Hugging Face's replay study.

## Measured results (this code, run 4 Nov 2026)

| Check | Correct model | Regressed model |
|---|---|---|
| `forward()` runs, correct shape | yes, `(14, 24)` | yes, `(14, 24)` — a smoke test passes it |
| Causality: max past-logit change | 0.000000 — pass | **1.025004 — FAIL** |
| Label log-prob drift vs baseline (−3.4498) | 0.0000 — pass | **0.2317 — FAIL** |
| Loss drift vs baseline (3.4498) | 0.0000 — pass | **0.2317 — FAIL** |
| Whole contract | PASS | FAIL |

Replay: commits 1, 2 PASS → **3, 4 flagged** → 5 PASS after the fix. Exactly the regressed commits, no others.

Fixture economics, same arithmetic as the proposal: the fixture is **28,032 params, 0.056 MB at BF16**, vs **70,859,776 params, 141.7 MB** for the full-size reference config in the file — about **2,528× smaller**, so the contract runs in well under a second on CPU and could plausibly run on every pull request.

## Why this matters for production ML work

- **Shape tests are not behaviour tests.** The regressed model returns finite logits of exactly the right shape. Only a probe that encodes an *invariant* (the future must not affect the past) or a *reviewed baseline* sees the break.
- **Baselines need a human in the loop.** A drift threshold is only meaningful if someone reviewed the reference output once; after that, the check is mechanical and cheap.
- **Replay turns one bug into a permanent guard.** Once a contract exists, replaying it over history (and then over every future change) converts a one-off postmortem into a standing regression test.

## Files

- `mic_harness.py` — TinyDecoder (causal and leaky variants), fixture sizing, contract build/validate, the three probes, commit replay, and a printed demo.
- `test_mic_harness.py` — 15 unit tests, all passing: fixture sizing, contract validation (missing baseline/example are flagged), causality, baseline and loss drift, the leaky model still running cleanly, and replay flagging exactly commits 3–4.

## Run it

```bash
pip install numpy
python3 mic_harness.py
python3 -m unittest test_mic_harness -v
```

## Sources

- Hugging Face, "Transformers Model Integration Contracts" (surfaced 3 Nov 2026), as reported at https://digitechbytes.com/technology-news-gadgets/transformer-model-integration-contracts-core-concepts-and-practices/ and https://www.aiwins.news/story/hugging-face-improves-how-new-ai-models-join-transformers-4ba8d3 — all Hugging Face figures above are theirs and are labelled as reported, not independently verified here.
