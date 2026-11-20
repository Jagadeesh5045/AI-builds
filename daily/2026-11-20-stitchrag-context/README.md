# StitchRAG in miniature — compact, evidence-preserving RAG contexts

**Trending source:** "StitchRAG: Constructing Compact and
Evidence-Preserving Contexts for RAG with Diffusion-Based Stitching"
(Kim, Lim, Cho & Baek, arXiv:2611.22766, 18 Nov 2026, as reported by
Grafy News). The paper's claim, in its own framing: instead of feeding
an LLM the raw retrieved passages, compress each passage and choose
sizes by **utility maximisation** — the most answer-bearing
information per token — so the context shrinks while factual accuracy
holds and latency falls with the token count. The compression itself
is done by a learned diffusion model; the reported token savings and
accuracy figures are the paper's, on its benchmarks.

## What this demo does

A stdlib-only, deterministic miniature of the *decision* StitchRAG
makes, with the learned diffusion generator replaced by an explicit
greedy optimiser you can inspect line by line:

- A 6-passage support knowledge base about one fictional checkout
  outage. Passages deliberately overlap: the same facts are repeated
  in near-identical words across passages, the way real top-k
  retrieval wastes context.
- A keyword retriever returns all 6 passages for one question.
- **Naive baseline:** concatenate the passages whole — 250 tokens.
- **Stitching:** every sentence is a candidate stitch with a utility
  (new query terms covered, gold-fact terms weighted double) and a
  cost (tokens). Stitches are picked greedily by utility per token
  under ONE global budget across all passages, near-duplicates of an
  already-kept stitch are skipped, and a sentence is kept whole or
  dropped whole — evidence is never truncated mid-fact.

## Results (run it: `python3 stitchrag.py`)

| Context | Tokens | Gold-fact recall | Redundancy | Recall / 100 tokens |
|---|---|---|---|---|
| Naive top-k | 250 | 1.000 | 0.125 | 0.400 |
| Stitched, budget 60 | 58 (−77%) | 0.875 | 0.000 | 1.509 |
| Stitched, budget 90 | 72 (−71%) | 1.000 | 0.000 | 1.389 |
| Stitched, budget 120 | 72 (−71%) | 1.000 | 0.000 | 1.389 |

Three honest observations:

1. **The win is redundancy, not summarisation.** Nothing was
   rewritten; the stitched context is 7 whole sentences from 4
   passages. Overlapping retrieval had quietly spent ~29% of the
   naive context re-stating facts.
2. **Budget past saturation buys nothing.** 90 and 120 give identical
   contexts: once every fact is covered, marginal utility is zero and
   the greedy stops. That is the paper's utility framing working as
   intended — and a warning against "just raise the budget" tuning.
3. **The recall metric is a proxy with a known flaw.** A fact counts
   as covered when all its key terms appear *somewhere* in the
   context, so numbers can collide across sentences (the "09" of the
   09:40 email plus the "12" of 12,400 checkouts can jointly
   "cover" the 09:12 start time even if that sentence was dropped).
   Treat the recall figures as an upper bound; a production version
   needs per-sentence fact attribution, which the verbatim test
   (`test_evidence_preserved_verbatim`) only partly provides.

## What this is not

The paper's diffusion-based stitching *generates* compressed text and
can merge facts across passages; this toy only *selects* whole
sentences, so it cannot compress a single long evidence sentence, and
its utility weights (facts ×2) are hand-set rather than learned.
The point stands at toy scale: in RAG pipelines, context cost is
often a selection problem before it is a generation problem.

## Files

- `stitchrag.py` — corpus, retriever, naive baseline, greedy stitcher,
  fact-recall and redundancy metrics, demo `main()`.
- `test_stitchrag.py` — 12 tests: budget respected at 4 budgets,
  verbatim evidence preservation, zero near-duplicate stitches,
  determinism, per-token efficiency beating the naive baseline.
  All 12 passing.

Run: `python3 stitchrag.py && python3 test_stitchrag.py`
