# MEMOIR-style memory lifecycle: revise importance after use, certify merges before they land

**Trending:** *MEMOIR: A Training-Free Memory Lifecycle for Long-Horizon LLM Agents* (Anand, Raj, Jha & Saha, [arXiv:2611.07631](https://arxiv.org/abs/2611.07631v1), submitted 5 Nov 2026), covered in the 7 Nov AI overnight briefings. MEMOIR stands for *Memory Evidence via Merge-certified, Online Importance Revision*.

The paper's complaint is one most agent builders will recognise: memory systems write each memory with a **static, prospective importance score and never revise it after the memory is used**. MEMOIR closes the loop:

1. **Retrospective attribution** — after each query, estimate which retrieved memories actually supported the answer and raise their retrieval priority. The update is **confidence-gated**, so it needs no reference answers and no reranker training.
2. **Merge-certified consolidation** — before memories are compressed into one, a screening gate checks that answer-relevant evidence survives; uncertified merges roll back.

The paper's own figures (theirs, not measured here): on 1,905 deduplicated LoCoMo QA pairs the full stack improves evidence recall@5 by **11.92 points** over bare cosine; confidence-gated feedback alone adds **1.50 points**; a deployment-unavailable oracle adds 1.84 overall and 8.05 on multi-hop. Judge-gated merges changed text-evidence recall 58.82% → 59.08%, shrank the store by 1.42%, and had **0 observed losses in 100 audited merges** (one-sided 95% upper bound 2.95%).

## What this demo does

Pure stdlib Python, one file, fully deterministic. A templated store of **92 memories** about 8 people (36 fact memories including 4 paraphrase duplicates, 56 notes/chatter), token-cosine retrieval, and an extractive answerer. Attribution is **leave-one-out**: a retrieved memory earns a credit when removing it changes the answer or drops the winning score by ≥25% — but only when the answerer's confidence clears the gate (winner's share ≥ 0.45). Credits feed a saturating priority, `1 − exp(−0.5·credits)`. The stack also adds bridge expansion (one hop from the best person-matching memory along shared project tokens) and a small exploration bonus for never-retrieved memories.

A learning session of **43 queries** (40 real, 3 deliberately ambiguous), then a 40-question evaluation set (48 evidence slots) with paraphrase and multi-hop questions, including a lexical-gap family — *"What is Maya's pastime?"* shares no content token with the hobby memory beyond her name.

## Measured here (not borrowed)

| mode | recall@5 | recall@3 |
|---|---|---|
| bare cosine | 0.833 | 0.688 |
| stack, no feedback | 0.833 | 0.833 |
| full MEMOIR-style | **1.000** | 0.833 |

- Session: 40/43 answered confidently, **exactly the 3 ambiguous queries gated out** (no update), 48 attribution credits applied, and **0 credits to any note/chatter memory** — leave-one-out attribution spares them because removing them never changes an answer.
- The components split the work differently at different k, which the table shows honestly: **bridge expansion owns the @3 gain** (it rescues multi-hop evidence), while **retrospective feedback owns the @5 gain** — the pastime evidence sits in a 4-way cosine tie for the last top-5 slots, and only the priority it earned in-session lifts it in. In the paper, feedback was the small slice (+1.50); in this toy it is the whole @5 slice. Small templated sets exaggerate whichever mechanism addresses their specific failure mode.

## Consolidation: similarity is not safety

112 high-similarity fact pairs were proposed. Compression here is token intersection (only what both phrasings agree on survives — a lossy summariser). A merge certifies only if every represented fact's **value token survives** *and* probe questions lose no evidence.

- **4 certified** — exactly the four paraphrase-duplicate pairs. Store 92 → 88 (−4.3%).
- **108 rejected** — including cross-person hobby pairs at cosine **0.50**, *higher* than the duplicates' 0.471: merging "enjoys tennis" with "enjoys cycling" keeps only {enjoy, weekend} and destroys both facts. The gate, not the similarity score, is what makes consolidation safe.
- Audit on the held-out evaluation questions: **0 of 48 evidence slots lost**. But with only 4 audited merges, the one-sided 95% upper loss-rate bound is **52.7%** — the code reproduces the paper's 2.95% exactly at n=100, which makes the point quantitative: small audits certify weakly, and the paper's n=100 is doing real work.

## Two bugs the tests caught while building

- The first stemmer did not converge *managed*/*manages*, so every multi-hop question silently lost its bridge token and gated out. Retrieval demos fail at the tokenizer more often than at the model.
- The first bridge-expansion rule also boosted person-mentioning notes, crowding gold evidence out of the top-3. Expansion now hops only to project memories that do not mention the person.

## Run it

```bash
python3 memoir_lifecycle.py          # demo with the tables above
python3 -m unittest test_memoir_lifecycle -v   # 17 tests
```

No LinkedIn post is auto-published from this run; this is a small daily experiment.
