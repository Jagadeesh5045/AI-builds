# Engram-style conditional memory: capacity without compute

A stdlib-only demo of the memory mechanism at the centre of Tencent's
Hy4 flash release: hashed n-gram lookup tables (Engram tables) fused
into the residual stream through a gate, so factual capacity can grow
without growing per-token compute.

## Why this, why now

On 17 November 2026 Tencent published the open weights of **Hy4 flash**
on Hugging Face in three forms (the instruction-tuned model, an FP8
build, and the pre-trained base), under the Tencent Hy License 1.0.
The architecture is the story: as reported at the release, the model
has **220B total parameters, of which about 64B are Engram tables**,
only **about 12B parameters are active per token**, and the context
window is **1,048,576 tokens**. (The launch material said 224B in one
place and the model card's own table sums to 220.3B; the licence also
carries a reported clause requiring a separate written licence for
model-as-a-service businesses above US$50M revenue or 100M monthly
users. All figures in this paragraph are the vendor's/reported ones,
not measurements made here.)

The Engram idea, introduced by DeepSeek in early 2026, is a split of
labour: transformer weights are good at composition and reasoning, but
a large share of stored knowledge is static factual association, and
that does not need matrix multiplies. It can live in a table addressed
by a hash of the local token n-gram, read in O(1), and gated into the
residual stream. Memory becomes a storage problem, not a compute
problem.

## What the code does

`engram_memory.py` models exactly that mechanism, with nothing trained
and nothing hidden:

- **`EngramTable`** — multi-head hashed n-gram tables (n = 2, 3). Each
  head uses an independent hash seed. A slot stores the mean of the
  value vectors written to it, so hash collisions show up honestly as
  averaged, corrupted memories. Storage is a sparse dict of written
  slots (a production table is dense on disk; the addressed set is
  what the mechanism depends on).
- **Factual-recall experiment** — 200 synthetic subject → object
  facts, each object a random unit vector in 32 dimensions (the
  hardest case: objects share no structure). Facts are recalled by
  nearest-neighbour decoding of the memory vector.
- **Gate** — a logistic on the memory vector's norm. Stored facts read
  back with norm near 1 and open the gate; unseen contexts read back
  the zero vector and the gate stays shut.
- **Accounting** — per-token lookup FLOPs are a function of heads,
  n-gram orders and width only. The slot count never appears.

## Measured results (seeded, reproducible)

Recall@1 over the 200 facts:

| heads | slots | recall@1 |
|------:|------:|---------:|
| 1 | 256 | 0.955 |
| 1 | 1,024 | 0.995 |
| 1 | 4,096 | 1.000 |
| 1 | 16,384 | 1.000 |
| 4 | 256 | 1.000 |
| 4 | 1,024 | 1.000 |

Two effects worth noting:

1. **Collisions degrade gracefully, then vanish.** With one head, a
   256-slot table loses about 1 fact in 20 to collision averaging; by
   4,096 slots the table is effectively collision-free for 200 facts.
   Cramming 200 facts into 64 slots (see the tests) drops recall below
   0.6 — capacity is a real constraint, just a cheap one.
2. **Heads are collision insurance.** Four independent hash functions
   collide on *different* facts, so averaging the four reads restores
   recall to 1.000 even at 256 slots. This is why real Engram designs
   use multiple heads per table.

Gate behaviour: a stored fact reads back with gate **0.953**; an
unseen subject gets 0 hits and gate **0.047** — the memory branch
stays silent when it holds nothing.

Compute accounting at heads = 4, orders = 2, width = 64: the lookup
costs **576 FLOPs per token** whether the table holds 1,000 slots
(0.00 GiB) or 64,000,000 slots (30.52 GiB). That constant is the whole
architectural argument, and it is why Hy4 flash can report 64B of
memory parameters alongside only ~12B active ones.

## Honest caveats

- No language model is trained here. The demo isolates the lookup,
  collision, gating and accounting behaviour; it says nothing about
  Hy4 flash's quality, which this run did not evaluate.
- Random unit-vector objects are deliberately structureless. Real
  learned embeddings cluster, which changes collision damage in both
  directions.
- The gate here keys on vector norm as a proxy for "the table holds
  something intact"; production Engram gating is learned and
  content-dependent.

## Run it

```bash
python3 engram_memory.py        # demo sweep + accounting
python3 test_engram_memory.py   # 13/13 tests, plain runner
python3 -m pytest test_engram_memory.py
```
