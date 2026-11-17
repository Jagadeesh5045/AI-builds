# Context-biased STT decoding: phrase boosting + formatting instructions

**Trending:** AssemblyAI released **STT-3.7**, **STT-3.7-Realtime** and
**STT-3.7-Realtime-Agent** on **17 November 2026** (per the November 2026
release tracker). The reported headline features are 32 languages with
automatic detection plus an optional 99-language fallback, **context
biasing**, and **natural-language formatting instructions**. Those figures
and feature descriptions are AssemblyAI's / the tracker's, not measured
here. This demo builds the two headline mechanisms from scratch so the
trade-offs are visible: how much should a decoder trust a list of expected
phrases, and what happens when it trusts them too much?

## What the code does

`stt_context_bias.py` is pure Python (stdlib only), fixed data, no audio
and no model. Each of 8 utterances is a small *lattice*: every position
offers candidate tokens with acoustic log-scores.

- **Baseline decoding** (bias strength β = 0) is per-position acoustic
  argmax via a Viterbi search. It mangles exactly the words a real
  recogniser mangles: rare domain terms that sound like common words.
  "pipeline pulse" → "pipe plus", "rag eval lab" → "rack evil lap",
  "systmone" → "system", "duckdb" → "duck". Mean WER **0.1952**.
- **Context biasing** adds a bonus β once per *completed* bias phrase
  (unigram, bigram or trigram, e.g. `pipeline pulse`, `rag eval lab`,
  `power bi`, `systmone`) appearing consecutively in the path. The DP
  state is the last two tokens, enough for phrases up to length 3. The
  bias can only re-rank candidates already in the lattice; it cannot
  invent tokens (there is a test for exactly that).
- **Bias-strength sweep** — the point of the demo:

  | β | mean WER |
  |---|---|
  | 0.0 | 0.1952 |
  | 0.25 | 0.1286 |
  | 0.5 | 0.1036 |
  | **0.75** | **0.0000** |
  | 1.0 | 0.0208 |
  | 1.5 | 0.0208 |
  | 2.0 | 0.0208 |

  A moderate β repairs every domain term. Past the optimum, WER rises
  again and never recovers: the control utterance "she rode a power
  **bike** home" (where no bias phrase is actually said) flips to "power
  **bi**" once β exceeds the 0.95 acoustic margin. That is the real
  production lesson of phrase boosting: the bias list is a prior, and an
  over-strong prior manufactures the very errors it was meant to fix.
- **Formatting instructions**: a tiny interpreter for plain-English
  instructions. "Capitalise product names" applies canonical casing
  (`duckdb` → `DuckDB`, `bi` → `BI`, `systmone` → `SystmOne`), "use the
  pound sign for amounts in pounds" rewrites `50 pounds` → `£50`, and
  "capitalise the first word" sentence-cases the output. An instruction
  that is not a formatting rule ("make it sound friendlier") is recorded
  as ignored rather than guessed at.

## Results (this demo's own numbers, reproduced by running it)

- Baseline mean WER 0.1952 → **0.0000 at β = 0.75** on the 8-utterance set
- Over-biasing (β ≥ 1.0) reintroduces error only on the control utterance:
  mean WER 0.0208, from the single manufactured "power bi" substitution
- Formatting: `the power bi dashboard uses duckdb` →
  `The power BI dashboard uses DuckDB`; `the invoice was 50 pounds` →
  `The invoice was £50`
- **16/16 tests passing**

## Run it

```bash
python3 stt_context_bias.py        # demo: baseline, sweep, formatting
python3 test_stt_context_bias.py   # 16 tests, no pytest needed
```

## Honest limits

The lattices are hand-built with deliberate acoustic margins (domain terms
0.15–0.30 below their confusables, the control confusable 0.95 below), so
the sweep's shape, not its exact β values, is the transferable result.
Real biased decoders (shallow fusion, WFST biasing, contextual adapters)
operate on frame-level acoustics and much larger lattices; the re-ranking
principle and the over-biasing failure mode are the same.
