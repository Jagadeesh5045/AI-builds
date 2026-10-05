# chunk-bench

A benchmark harness that measures **which document-chunking strategy retrieves best for RAG**.
Fix the retriever, vary the chunker, and get numbers instead of opinions.

## Problem

Every RAG team eventually argues about chunking: fixed-size or semantic? 500 or 2000
characters? Sentence-aware or heading-aware? Most teams pick by gut feel because measuring
the difference takes a bespoke eval setup. chunk-bench is that setup: point it at a corpus
and a set of gold questions, and it scores each strategy on retrieval quality.

## Real-world use case

You are building a support assistant over 2,000 help-centre articles. Before committing to a
chunking strategy in production, you run chunk-bench on a sample of 50 articles with 100
real support questions. The results tell you whether sentence-aware chunking at 1000 chars
beats fixed-size at 500, and what each choice costs you in index size. You ship the winner
with evidence, not a hunch.

## Approach

1. **Chunk** every document with each strategy in the grid (fixed-size at 500/1000/2000
   chars, sentence-aware at 1000/2000, heading-aware at 1000/2000).
2. **Index** each chunking with the *same* TF-IDF retriever, so metric differences are
   attributable to chunking alone, never to retrieval.
3. **Score** every gold question with answer-containment relevance: a chunk counts as
   relevant if it contains the question's gold answer span. Metrics are recall@k, MRR,
   and context precision@k, plus index stats (chunk count, average chunk size).

The bundled corpus has 12 short guides on RAG topics and 24 hand-written gold QA pairs.
Swap in your own `data/corpus/*.md` and `data/qa_gold.json` to benchmark your domain.

## How to run

```bash
pip install -r requirements.txt
python run_bench.py              # full grid, k=5
python run_bench.py --k 10       # looser cutoff
python -m unittest discover -s tests   # 20 unit tests
```

Results land in `results/benchmark_results.csv` and `results/summary.md`.

## Sample output

Real run, 12 documents, 24 questions, k=5, TF-IDF retrieval:

```
strategy        recall@k  mrr    context_precision@k  n_chunks  avg_chunk_chars
-------------------------------------------------------------------------------
fixed-1000      1.000     0.917  0.325                24        795
fixed-2000      1.000     0.958  0.200                12        1489
sentence-1000   1.000     0.958  0.325                24        803
sentence-2000   1.000     0.958  0.200                12        1482
fixed-500       0.958     0.799  0.433                46        425
heading-1000    0.958     0.922  0.425                49        362
heading-2000    0.958     0.922  0.425                49        362
```

What the numbers say: on short documents, large chunks trivially cover every answer
(recall 1.0) but dilute the index, so context precision drops. Smaller chunks retrieve
more precisely (precision up to 0.43) at the cost of occasional misses. Sentence-aware
chunking matches or beats fixed-size at the same budget without ever cutting a sentence
mid-thought. The honest takeaway for production: benchmark on *your* documents, because
the winner depends on document length and question style.

## Tech stack

Python, scikit-learn (TF-IDF retrieval), NumPy, Pandas. No API keys, no network calls,
no heavy dependencies. The sentence splitter and all chunkers are dependency-free pure
Python; the whole benchmark runs on a laptop in seconds.

## Project structure

```
chunk-bench/
  run_bench.py            # entry point: runs the grid, prints the table, writes results/
  chunk_bench/
    chunkers.py           # fixed-size, sentence-aware, heading-aware chunkers
    corpus.py             # markdown corpus loader
    retrieval.py          # TF-IDF retriever (held constant across strategies)
    metrics.py            # recall@k, MRR, context precision, answer containment
    qa.py                 # gold QA set loader + validation
    evaluate.py           # strategy grid, benchmark driver, table formatter
  data/
    corpus/               # 12 bundled RAG-topic documents
    qa_gold.json          # 24 hand-written questions with gold answer spans
  tests/                  # 20 unit tests, including gold-set validation
  results/                # sample run output (regenerate with run_bench.py)
```

## Limitations

Answer containment is a proxy, not a perfect relevance judgement; a chunk can contain the
answer span without fully answering the question. The bundled docs are short, so large
chunk sizes degenerate to whole-document retrieval. For a production decision, use longer
documents from your own domain and add LLM-judged relevance on a sampled subset.
