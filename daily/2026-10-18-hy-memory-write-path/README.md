# Hy Memory 2.0-style memory write path (toy demo)

**Trending:** on 16 Oct 2026, Tencent Hunyuan released **Hy Memory 2.0** —
an open-source agent-memory stack with three layers: the
**Hy-Memory-30B-A3B** model, the Hy-Memory framework, and the **MemHub**
local app (all Apache 2.0). The core thesis: stop treating all agent memory
as one undifferentiated chat log. Separate **working memory** from **code
memory** (and chat memory), and hand the *write path* — deciding what to
persist, where, and in what form — to a small dedicated specialist (~30B
total params, ~3B active per inference) instead of a general LLM. Tencent
trains it with Memory On-Policy Distillation plus attributed RL on memory
(per-chunk checks, rewards attributed across chunks; arXiv paper dated
15 Oct 2026). Tencent's own reported numbers (55.49% avg across four memory
frameworks vs Claude Opus 5's 54.62% and DeepSeek V4.1 Flash's 53.64%;
72.6% on LoCoMo) are **company claims, not independently verified**.

## What this demo implements

`hy_memory.py` — a toy *protocol* of that write path, in stdlib-only Python:

- **Scene routing** (`_classify_scene`): every turn is routed to one of three
  scenes — `chat`, `working`, `code` — mirroring Hy Memory's chat / working /
  coding memory split.
- **Salience scoring** (`_salience`): a heuristic stand-in for the learned
  write reward — checkable facts and reusable lessons score high, chit-chat
  is skipped at the write path (4 of 14 turns skipped in the demo session).
- **Supersede tombstoning** (`_supersede`): when a new chunk asserts a
  different value for the same (entity, attribute) pair, the old chunk is
  tombstoned, not deleted — so queries never return outdated facts. This is
  failure mode #1 a practitioner measured on real Hy Memory this week
  (answering with an outdated version of a changed fact).
- **Code diff-chains** (`_write_code`): every code version is stored with a
  unified diff *and the rationale for the change* — Hy Memory's
  version-evolution tracking: *why* it changed, not just *what*.

## Demo results

`demo.py` runs a 14-turn scripted session (fact updates, a reusable lesson,
three evolving code versions) and scores 6 strict queries — a query counts
only if the expected answer is present **and** the stale value is absent:

| system | strict accuracy | tokens written | tokens retrieved |
|---|---|---|---|
| HyMemory (3 scenes + supersede + diff-chain) | **6/6** | 154 | 212 |
| RawLog (one undifferentiated chat log) | 4/6 | 194 | 375 |
| FlatWriter (general-LLM stand-in, flat list) | 4/6 | 182 | 354 |

Both baselines fail the two outdated-fact queries (they return 5432 *and*
5433, Monday *and* Wednesday — grep cannot tell new from old), and retrieve
~1.8x more tokens per query. `test_hy_memory.py`: 18/18 passing.

## Honest caveats

- The salience/entity/diff heuristics are a **toy stand-in**, not
  Hy-Memory-30B-A3B. The real model *learns* the write protocol from data;
  this demo hard-codes six fact patterns and a keyword router.
- The demo itself surfaces a write-path redundancy the real model would
  learn away: the assistant's echo ("Staging database port 5432 recorded…")
  is written as a second chunk in the chat scene. Both copies are correctly
  tombstoned by the later update, but a learned writer would dedupe at write
  time.
- All accuracy/token numbers are from the scripted toy session above, not
  from LongMemEval/LoCoMo or any real benchmark.

## Files

- `hy_memory.py` — write path: `MemoryWriter`, scene routing, salience,
  supersede tombstoning, code diff-chains, scene-targeted retrieval, plus
  `RawLog` and `FlatWriter` baselines.
- `demo.py` — scripted session + strict scoreboard. Run: `python3 demo.py`
- `test_hy_memory.py` — 18 unit tests. Run: `python3 -m unittest test_hy_memory -v`
