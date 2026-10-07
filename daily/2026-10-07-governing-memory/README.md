# Governing Facts vs Episodic Facts (2026-10-07)

A working demo of this week's finding from AI memory research: **similarity
search is the wrong retrieval mechanism for an agent's standing rules**.

## The trending hook

A study reported on 7 October 2026 by Oghenenefe Abeke, Rasheed Mohammad and
Haitham Mahmoud (Birmingham City University), published in the *International
Journal of Data Science and Analytics*, draws a sharp line between two kinds
of facts a conversational AI accumulates:

- **Episodic facts** — "we debugged the login flow on Tuesday". These *answer*
  a query, so top-k similarity retrieval is the right tool.
- **Governing facts** — "never use pandas in this project; use polars instead".
  These do not answer a query; they *bind every query*. Their force is
  constitutive rather than topical, and that is where the dominant
  retrieve-by-similarity architecture breaks down.

The motivating example: you tell your coding assistant on Monday "never use
pandas; use polars". All week the conversation drifts through logging,
refactoring, storage. On Friday you ask for a dedup utility, and the assistant
cheerfully writes `import pandas`. The instruction was stored, indexed, and
simply never retrieved.

## What this code shows

`governing_memory.py` implements the failure mode and the fix in pure Python
(numpy for cosine similarity; hashed TF vectors stand in for real embeddings):

1. `NaiveAgent` — one episodic memory, top-k similarity retrieval, no
   governing layer. Friday's query ("write a quick utility to deduplicate some
   rows") has **cosine 0.000** with Monday's rule but 0.204 with unrelated
   chatter, so the rule falls out of the top-3 context and the agent writes
   pandas code. The failure is structural, not embedding quality.
2. `GovernedAgent` — same episodic store **plus a `ConstraintStore`** that is
   consulted on *every* turn. Standing rules ("never/always/do not use …") are
   parsed out of user messages, the planned action is checked against every
   rule before execution, violations are blocked and the agent regenerates a
   compliant action (pandas → polars).

Run it: `python3 governing_memory.py` (asserts the failure mode reproduces and
the governing layer fixes it). `test_governing_memory.py` holds 10 tests —
all passing.

## Results

- Naive agent: rule buried at rank >3, `import pandas` executed. Rule broken.
- Governed agent: violation flagged (`prohibited_found: ['pandas']`), action
  regenerated with polars. Rule honoured.

## Honest scope

This is an independent, simplified implementation of the paper's core idea —
it is not a reproduction of the paper's experiments. The constraint parser is
a heuristic (never/always/do-not patterns); production systems would extract
rules with a classifier or LLM and enforce them in the tool-call layer.

## Source of the finding

scienmag.com, 7 Oct 2026: "AI Assistants Forget Your Rules Because Similarity
Search Is the Wrong Tool" — reporting the Birmingham City University paper.
