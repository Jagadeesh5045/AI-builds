# Agent Context Compression (Headroom-style)

**Date:** 2026-10-10 · **Trending:** agent context compression — the open-source
"Headroom" layer (headroomlabs-ai/headroom, tens of thousands of GitHub stars,
covered again this week) claims 20% fewer tokens for coding agents and 60–95%
for JSON-heavy workflows by compressing what the agent *reads* before it reaches
the model; Anthropic's Haiku 5.5 launch (Oct 7) has the whole agentic-tooling
crowd talking about cheaper inference again.

## What this shows

Production agents hit context limits fast: a coding agent that runs tests,
reads logs and pulls docs can burn 100k tokens in three turns, and every token
costs money and latency. This demo implements the core idea of a compression
layer that sits between the agent and the LLM — shrink tool outputs, logs, RAG
chunks and files, but never the user query or the answer — using transparent,
deterministic heuristics instead of a learned compressor:

| Component | What it does | Invariant |
|---|---|---|
| `LogCompressor` | Keeps ERROR/FATAL/WARNING lines byte-for-byte; collapses repeated INFO/DEBUG; truncates middle noise with a summary marker | Severe lines always survive |
| `JsonCompressor` | Drops verbose keys (`trace`, `debug`, `metadata`...), truncates long strings/lists with visible markers | Output stays valid JSON, same shape for kept keys |
| `ChunkCompressor` | Sentence-level RAG filtering: scores sentences by query-term overlap, demotes boilerplate, keeps top sentences under a token budget, original order preserved | Answer-bearing sentences win |
| `AgentContextCompressor` | Orchestrates per-type compression under a token budget and reports tokens before/after, compression ratio, % reduction and estimated cost saving | — |

## Run it

```bash
python3 test_compression.py   # 18 unit tests
python3 demo.py               # realistic 3-turn agent trace, before/after stats
```

Token counts are estimated as `ceil(chars / 4)` (labelled as estimates; use
tiktoken in production). Cost figures use an illustrative $1.00 / 1M input
tokens, roughly Haiku-class pricing in Oct 2026.

## Files

- `agent_context.py` — the compression layer (stdlib only, no dependencies)
- `demo.py` — realistic agent trace: verbose test logs + tool JSON + RAG chunk
- `test_compression.py` — 18 unit tests covering the invariants above

## Limits (honest)

- These are heuristics, not a learned compressor: aggressive settings can drop
  useful nuance, so a real deployment would validate compression against the
  team's own task-accuracy metrics (the Headroom authors say the same).
- The "answer unchanged" check here is keyword-recall on the compressed
  context, a proxy — the same honest-proxy pattern as my other daily demos.
