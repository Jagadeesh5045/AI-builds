# K-EXAONE 2.5 context planner — budgeting for always-on reasoning

**Trending:** LG AI Research released K-EXAONE 2.5 on **9 November 2026**,
announced at NVIDIA AI Day Seoul and offered **free** through the new
EXAONE API Platform. All figures in this section are LG's / the platform
docs', vendor-reported and not independently verified here:

- Technology-validation model, further-trained from the August
  K-EXAONE release; 750B total / 31B active parameters, text only,
  Korean and English, January 2026 knowledge cutoff
- **524,288-token combined limit:** input tokens + `max_tokens` must
  not exceed 524,288, or the API returns HTTP 400
- **Reasoning is always on:** `reasoning_effort` is `low` / `high` /
  `max` (default `max`), it cannot be switched off, and reasoning
  tokens count toward `max_tokens`
- OpenAI-compatible `POST /v1/chat/completions`, model id
  `k-exaone_v2.5`, with tool calling, structured outputs and streaming
- LG reports +70% across 24 internal metrics vs the August model, and
  AutomationBench-Openset 74.6, SWE-Bench Pro 59.2, BrowseComp 78.0,
  Claw-Eval 77.3 (internal runs)
- Free at launch; no quota, rate limit or end date published

Sources: LLM Releases changelog (9 Nov 2026), Unite.AI, The Korea Times
via Digital Today, EXAONE API Platform documentation summaries.

## The problem this demo solves

Optional-thinking models let you budget context first and reasoning
later. An always-on model inverts that: reasoning is a fixed cost you
must reserve **first**, inside the same ceiling as your input. Forget
it and the request either fails with a 400 or the visible answer is
starved by the reasoning that runs before it.

`context_planner.py` (stdlib only) is the client-side planner:

1. **Reserve reasoning first** — illustrative reserves for this demo
   (LG publishes effort names, not per-tier token counts, so these are
   the planner's own policy): low 2,048, high 16,384, max 65,536
2. **Reserve the answer** you actually want back
3. **Pack context into what remains** — chunks scored by stopword-aware
   term overlap with the query, packed greedily in relevance order,
   never split mid-chunk; zero-relevance chunks are dropped as
   irrelevant even when space remains
4. **Validate the exact rule** — `input + max_tokens <= 524,288`,
   raising a 400-mirroring `ContextOverflowError` otherwise;
   `reasoning_effort="off"` is rejected outright

It also builds the OpenAI-compatible payload (`model`,
`reasoning_effort`, `max_tokens`, messages) a caller would send.

## Verified demo run

Synthetic analytics corpus: 20 chunks (1,624 estimated tokens) — five
regional Q3 sales notes, one returns deep-dive, an HR picnic note, a
weather note and 12 archive filler chunks. Query: *"Which region had
revenue growth in Q3 and why did returns rise in the South?"*

| Effort | Reasoning reserve | max_tokens | Input | Total | Kept / dropped |
|---|---|---|---|---|---|
| low | 2,048 | 3,072 | 192 | 3,264 | 6 / 14 |
| high | 16,384 | 17,408 | 192 | 17,600 | 6 / 14 |
| max (default) | 65,536 | 66,560 | 192 | 66,752 | 6 / 14 |

- The planner keeps **exactly the 6 relevant chunks** (5 sales notes +
  the returns deep-dive, top-ranked `sales-south`) and drops all 14
  irrelevant ones at every effort level
- Switching low → max costs **63,488 extra tokens of ceiling** for the
  same input and the same 1,024-token answer reserve: on an always-on
  model, effort choice *is* context choice
- Tight-ceiling run (limit 3,000, low, 256-token answer): total 2,496,
  the same 6 chunks kept — and a 600-token ceiling cannot be planned
  at all, because the low reasoning reserve alone exceeds it. That
  floor is a real property of always-on reasoning, caught by a test
- Token counts use a deterministic estimator in the demo, not LG's
  tokeniser; the arithmetic and invariants are the point

## Tests

16/16 passing (`python3 -m unittest`): token estimator determinism,
effort parsing (default max, `off`/`none`/`disabled` rejected, tiers
monotonic), plans always fitting the real 524,288 limit, reasoning
counting toward the budget, the exact-boundary overflow, relevance
ranking (returns deep-dive first for a returns query, zero for an
unrelated chunk), tight-limit dropping, no chunk ever split, the
tiny-limit always-on floor, and the payload shape.

## Run it

```bash
python3 context_planner.py
python3 -m unittest -v
```

No API key, no network, no model download: this is the planning layer
you need *before* calling the free API, not a client for it.
