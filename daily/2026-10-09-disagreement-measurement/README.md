# Disagreement measurement: deterministic verifier vs judge

An experiment in citation grounding for RAG systems. It runs two guards over
the same draft answers and counts where they disagree.

## Why this exists

On my Medium article about RAG evals, Nueravi ("Decoding AI") proposed this
protocol: implement both a deterministic citation line-check and an
LLM-as-judge, run them over the same drafts, and count the disagreements.
"Every case where the judge passes a citation the file does not support is a
judge error you can finally see." I replied that I would keep both guards.
This repo is the implementation. Thread:
https://medium.com/@padalajagadeesh578/keep-both-they-catch-different-failures-c9c3b74e86b0

## What is here

- `verifier.py`: deterministic citation verifier. Parses `path.py:LINE`
  citations, opens the file, checks the line exists, and requires at least
  50% of the claim's content tokens to appear on the cited line. Output:
  SUPPORTED / UNSUPPORTED / MALFORMED with reasons. No model calls.
- `judge.py`: the judge side. **Heuristic proxy, not an LLM.** No API key
  was available in this environment, so the judge is a TF-IDF cosine
  similarity between the claim and the cited line (threshold 0.35). It is
  labeled `proxy` in the code, in the results, and here. Do not read these
  numbers as LLM-judge results.
- `experiment.py`: 20 draft answers with a mix of good citations, wrong-line
  citations, hallucinated files, malformed citations, and borderline
  paraphrases. Runs both guards, prints the comparison, writes
  `results.json`.
- `fixtures/`: a small demo codebase (auth, retrieval, eval, db helpers)
  with known content that the drafts cite.
- `test_verifier.py`: 19 unit tests for the verifier. All pass.

## The numbers

- Total citations: 20
- Judge (proxy) pass rate: 11/20
- Verifier pass rate: 8/20
- Disagreements: 3
- Judge-passed-but-verifier-rejected: 3

All three disagreements are the same failure mode: the proxy judge passed a
citation the verifier rejected. Example: the claim "session tokens embed the
username between fixed markers" cites `fixtures/auth.py:21`
(`return "session-" + username + "-active"`). The line shares the rare terms
"session" and "username", so the proxy scores cosine 0.658 and passes it, but
only a third of the claim's tokens appear on the line, so the verifier
rejects it at 0.33 overlap. The claim is a loose paraphrase, not a grounded
statement, and the deterministic check is the only thing that says so.

## Interpretation

The verifier is the cheap, exact first pass. The judge covers claims a line
match cannot settle, but it can be confidently wrong about grounding, and
that wrongness is invisible without the line check. The disagreement count
is the measurement most eval tooling cannot produce: a visible, countable
set of judge errors. Keep both guards, run both, count the gap.

## Honest limitations

- The judge here is a TF-IDF proxy, not an LLM. A real LLM judge would
  disagree in different places and probably fewer of them. The protocol is
  the point; rerun with an API key to get LLM numbers.
- The verifier's 50% token-overlap threshold is a blunt instrument. It
  rejects good paraphrases and passes claims that share words without
  sharing meaning. It is a floor, not a ceiling.
- The fixtures are tiny and the drafts are hand-written. This demonstrates
  the measurement, it does not benchmark judges in general.
