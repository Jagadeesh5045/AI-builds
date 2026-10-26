# Where RAG loses the answer: five loss points, measured

26 October 2026. Inspired by the Towards AI piece published today,
"Your RAG Demo Works Because You Already Know the Answers": a demo RAG
system looks perfect when the person asking already knows which document
holds the answer, while a production system loses the same answer in
several distinct places. This experiment traces 10 questions over a small
synthetic corpus (6 policy/shipping documents) through those five loss
points, for a naive pipeline and a fixed pipeline, and measures survival
at each stage.

## The five loss points

1. **Chunking** - a fixed 120-character window cuts an answer sentence
   in two, so no single chunk contains the whole answer.
2. **Vocabulary** - the question words do not match the document words
   ("return window" vs "refund period"), so lexical retrieval ranks the
   gold chunk low.
3. **Top-k** - the gold chunk ranks below the retrieval cutoff (k=3).
4. **Context** - a lookalike chunk (same generic sentence, different
   product, different value) reaches the reader first, and the reader
   takes the wrong value.
5. **Generator** - with no gold chunk in context, an ungrounded reader
   still answers confidently instead of abstaining: a hallucination.

## The fixes (cheap, in order)

Structure-aware chunking (split on headings, never split a sentence),
metadata prepended to every chunk (document title + section, the
no-LLM-call version of Anthropic's contextual retrieval), synonym
expansion at query time, a larger retrieval set (k=8) with a bigram
rerank, and a context-grounded reader that abstains when the gold chunk
is absent.

## Measured results (run it yourself)

```
Questions: 10
Naive chunks: 7 | Fixed chunks: 7
1. gold sentence intact in one chunk: naive 9/10 | fixed 10/10
2-3. gold chunk in retrieval top-k: naive 7/10 | fixed 10/10
naive reader outcomes: {'correct': 6, 'wrong-lookalike': 1, 'hallucinated': 3}
fixed reader outcomes: {'correct': 10}
```

So the naive pipeline ends at 6/10 correct (1 wrong lookalike value, 3
confident wrong answers), while the fixed pipeline ends at 10/10 with no
hallucinations. The single biggest remaining naive failure class is the
generator answering without evidence, not retrieval itself: 7/10 gold
chunks are retrieved, but only 6/10 answers are right.

## Honesty notes

- No LLM is called. Retrieval is TF-IDF cosine over the chunks, and the
  reader is a small rule-based stand-in whose behaviour is documented in
  `rag_failure_modes.py` (it never peeks at the gold answer to decide).
  The numbers are therefore a clean demonstration of the mechanisms,
  not a claim about any production system or model.
- The corpus is synthetic and tiny, and the two pipelines share the same
  chunk count by coincidence (short documents). The point is the
  per-stage diagnosis, which is why the script prints the outcome for
  every question, not just the totals.

## Run

```bash
python3 rag_failure_modes.py
python3 -m unittest test_rag_failure_modes -v
```

Files: `rag_failure_modes.py` (experiment), `test_rag_failure_modes.py`
(6 tests), this README. Python 3 standard library only.
