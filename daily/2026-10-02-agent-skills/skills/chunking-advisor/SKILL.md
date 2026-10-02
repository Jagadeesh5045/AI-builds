---
name: chunking-advisor
description: Recommends document chunking strategies for RAG pipelines — chunk size, overlap, and splitter choice for dense, BM25, and hybrid retrieval. Use when building or tuning retrieval-augmented generation indexing, or when asked about chunk sizes, overlap, splitters, or chunking evaluation.
license: MIT
metadata:
  author: Jagadeeswara Rao Padala
  version: "1.0"
  category: rag
compatibility: any agent; scripts require Python 3.10+
---

# Chunking Advisor

## Overview
Advise on how to split documents into chunks for a RAG index. Bad chunking is the
most common silent failure in naive RAG: too large and the retriever returns
diluted passages; too small and semantic context is destroyed.

## Decision rules
1. **Default start:** recursive character splitting, 512 tokens, 10–20% overlap.
   Simple, predictable, and hard to beat as a baseline.
2. **Dense (embedding) retrieval:** 256–512 tokens. Embedding models compress rare
   tokens, so keep chunks semantically self-contained.
3. **BM25 / keyword retrieval:** 512–1024 tokens. BM25 rewards longer passages with
   more term matches (error codes, identifiers, citations).
4. **Hybrid (dense + BM25 with RRF fusion):** 512 tokens with 15% overlap — the
   production default that captures both signal types.
5. **Tables / code:** never split mid-row or mid-function; use structure-aware
   splitting (see `references/recursive-chunking.md`).
6. **Evaluation first:** run `scripts/chunk_stats.py` on a sample corpus to measure
   mean/median chunk sizes and overlap before committing to settings. The script's
   code never enters the agent's context — only its output does (Tier 3).

## Anti-patterns
- Fixed-size splitting without overlap on narrative text (destroys sentence context).
- Raising top-k beyond ~20 to compensate for bad chunks — the "lost in the middle"
  effect means middle chunks are effectively ignored.
- Semantic chunking as a first choice: expensive, and benchmarks show simple
  recursive splitting beats it on end-to-end accuracy.
