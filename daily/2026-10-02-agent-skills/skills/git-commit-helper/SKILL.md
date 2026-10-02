---
name: git-commit-helper
description: Writes clear conventional-commit messages (feat/fix/docs/refactor/test/chore) with a short imperative subject line and bullet body. Use when asked to write, improve, or review a git commit message.
license: MIT
metadata:
  author: Jagadeeswara Rao Padala
  version: "1.0"
---

# Git Commit Helper

## Format
`<type>(<scope>): <imperative subject under 72 chars>`

Types: feat, fix, docs, refactor, test, chore, perf, ci.

## Rules
1. Subject in imperative mood: "add", not "added" or "adds".
2. Body explains WHY, not what (the diff shows what). Wrap at 72 chars.
3. One logical change per commit. Reference issue numbers in the footer.

## Example
```
feat(rag): add hybrid BM25+dense retriever with RRF fusion

Dense retrieval missed exact-match queries for error codes while BM25
missed paraphrases; RRF fuses both ranked lists without calibration.

Closes #42
```
