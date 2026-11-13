# Query fan-out planner — one search per sub-question, with a coverage report

**Trending (13 Nov 2026):** Google released **Gemini 4 Hydrogen** today, a fast
agentic model (Google reports a 51 Artificial Analysis score and 57% on
Terminal-Bench 4.1, with a 1M-token window and four thinking levels — Google's
figures, via MarkTechPost's 13 Nov release roundup). Rolled into Search / AI
Mode, Google highlights two things for complicated questions: better handling
of multi-part questions and enhanced reasoning for **query fan-out** — split
the tangled question into sub-queries, search for each one, then synthesise
an answer that covers every part (AI Daily / Rocket News coverage, 13 Nov).

This demo rebuilds that mechanism in pure stdlib Python over a six-document
corpus, so every step is inspectable. No model, no network, no dependencies.

## What the code shows

`query_fanout.py`:

1. **Decompose** — transparent rules split a complex question into
   sub-questions (semicolons; ", and <interrogative>"; " and
   <interrogative>"; comma lists). A plain "and" joining nouns ("cost of
   storage and compute") does **not** split. A simple question stays one
   sub-query: no fan-out is spent where there is nothing to fan out.
2. **Retrieve per part** — a small TF-IDF cosine index (stdlib `math`/`re`)
   is searched once per sub-question, so every facet gets its own retrieval
   budget instead of competing inside one diluted whole-question query.
3. **Coverage report** — each sub-question is marked `covered` (best score
   ≥ 0.20), `weak` (≥ 0.08) or `missing` from its own best score.
4. **Synthesise honestly** — the answer skeleton cites one document per
   sub-question; a part with no evidence is stated as unanswered or weak,
   never filled in by guessing.
5. **Single-shot baseline** — one search with the whole question, for an
   honest comparison.

## Measured on the built-in demo (verified by running it)

Corpus: six facet documents about a warehouse migration (cost, latency,
data quality, rollback, security, timeline).

| Question | Fan-out result |
|---|---|
| "What does the warehouse migration cost each month, how fast will dashboards load, and how do we roll back if the cutover fails?" | 3 sub-questions, **coverage 100%** — cost 0.42, latency 0.32, rollback 0.44 |
| "Which data quality checks run before load and who can access restricted rows?" | 2 sub-questions, **coverage 100%** — data_quality 0.70, security 0.45 |
| "What is the monthly cost of the warehouse?" | 1 sub-question (no fan-out), covered, 0.45 |
| "What does the warehouse cost each month and what training is provided for executives on presentation skills?" | cost covered 0.45; the executive-training part scores only **0.12 (weak)** — the corpus only mentions analyst training in week five, so the report flags a near-miss instead of claiming an answer. Coverage 50% |

**Honest baseline note:** on this six-document corpus, the single-shot
baseline's top-3 also happens to contain all three facet documents for the
first question — with six documents, top-3 is half the corpus. Fan-out's
advantage here is structural, and it is the part that scales: each facet is
*guaranteed* its own best-match slot and its own coverage verdict, whatever
the corpus size, and a missing facet surfaces in the report rather than
being silently dropped from a crowded top-k. In a large corpus, the diluted
whole-question query is exactly where facets fall out of the top-k.

## Why this matters for data / RAG work

Multi-part questions are the normal shape of stakeholder questions ("what
did it cost, how fast is it, and what happens if it breaks?"). A coverage
report per sub-question turns synthesis from "write something plausible"
into "answer these parts, flag those parts" — the same discipline as a
data-quality gate: measure each part, quarantine what has no evidence.

## Run it

```bash
python3 query_fanout.py            # demo: 4 questions, coverage per part
python3 -m pytest test_query_fanout.py   # 15 tests
```

15/15 tests passing: decomposition (simple, 3-way, semicolon, noun-"and"
non-split), retrieval per facet, fan-out coverage, missing/weak flagging,
and the baseline comparison.
