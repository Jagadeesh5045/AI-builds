# groundcheck

A groundedness and citation checker for retrieval-augmented generation (RAG) answers. Give it the passages a model retrieved and the answer it generated, and groundcheck tells you which claims in that answer are actually supported, which are not, and whether the answer should ship, ship with a warning, or be withheld.

## Problem statement

RAG systems still hallucinate. The model retrieves the right policy page and then states a wrong price, an invented feature, or a limit that was never in the source. Teams usually catch this by reading answers by hand, which does not scale. groundcheck is the automated first pass: a fast, deterministic checker that runs before an answer reaches a user.

## Real-world use case

A support assistant answers billing and account questions from a help centre. Before any answer is sent, groundcheck splits it into claims, checks each claim against the retrieved passages, verifies that numbers match the source exactly, checks that citations point at real passages, and flags negation flips (a source that says a link expires, an answer that says it never expires). Unsupported answers are routed to a human or regenerated instead of being sent.

## Approach

1. Claim splitting: sentence splitting, plus clause splitting on coordinating conjunctions when both sides are substantial, because one sentence often carries one grounded claim and one invented claim.
2. Support scoring: each claim is scored against every passage with token coverage (70%) and bigram coverage (30%).
3. Number matching: every number in a claim must appear in its best matching passage. A wrong price fails even when the wording is nearly identical.
4. Negation check: negation and absolutes in a claim (never, unlimited, instantly, free) must also appear in the source passage.
5. Citation validation: inline citations like [1] must reference an existing passage.
6. Decision policy: groundedness 1.0 answers, 0.6 to 0.99 answers with a warning, below 0.6 abstains.

Pure Python standard library. No LLM calls, no network, deterministic and unit tested, so it can run in CI or as a cheap pre-send gate.

## Results (seeded evaluation, reproduced by run_demo.py)

Eight labelled cases over six support passages: four fully grounded answers and four answers with planted unsupported claims (wrong price, never expiring links, instant cash refunds, an invented video editor feature).

- Grounded answers: 4 of 4 passed with groundedness 1.0
- Hallucinated answers: 4 of 4 flagged, decision abstain
- Detection precision 1.0, recall 1.0 (TP 4, FP 0, FN 0, TN 4)
- Tests: 6 of 6 passing (python3 -m unittest discover -s tests)

The first version scored recall 0.5: two hallucinations hid inside mostly correct sentences. Clause level splitting and the negation check fixed both, which is the honest lesson of the build: sentence level checks are too coarse for real generated answers.

## How to run

```bash
python3 run_demo.py            # writes report.md and prints the evaluation table
python3 -m unittest discover -s tests
```

```python
from groundcheck.checker import check_answer
report = check_answer("The Pro plan costs 9 pounds per month [1].",
                      ["The Pro plan costs 29 pounds per month and includes 10,000 API calls."])
print(report.decision, report.unsupported)  # abstain, 1 (the price is wrong)
```

## Tech stack

Python 3.10+ standard library only (re, dataclasses, unittest). No third party dependencies.

## Layout

- `groundcheck/checker.py` claim splitting, scoring, number, negation and citation checks, decision policy
- `groundcheck/data.py` six passage support knowledge base and eight labelled cases
- `groundcheck/evaluate.py` evaluation harness with precision and recall
- `run_demo.py` demo entry point that generates report.md
- `tests/test_checker.py` six unit tests
