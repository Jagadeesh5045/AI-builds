# Recursive chunking — Tier 3 reference (loaded on demand)

> This document is loaded only when the skill's instructions point to it.
> It never sits in the agent's startup context.

## Algorithm
Recursive character splitting tries a prioritized list of separators and falls
back to finer ones when a piece still exceeds the target size:

1. `\n\n` (paragraphs)
2. `\n` (lines)
3. `. ` / `? ` / `! ` (sentences)
4. ` ` (words)
5. `` (characters — last resort)

Each chunk is then extended with `overlap` characters from the previous chunk so
boundary sentences keep their context.

## Worked example
Target 512 chars, overlap 75 chars, separators `["\n\n", "\n", " ", ""]`:

- Document split on `\n\n` → 3 paragraphs of 400/700/200 chars.
- The 700-char paragraph is re-split on `\n`, then on `" "` until every piece
  fits within 512 chars.
- Each chunk keeps the last 75 chars of its predecessor as overlap.

## When to reach for semantic chunking instead
Only after the recursive baseline is measured and found wanting on your
retrieval metrics (recall@k on a labelled eval set). Cost per document is
10–100x higher (an embedding call per candidate boundary), so demand evidence.
