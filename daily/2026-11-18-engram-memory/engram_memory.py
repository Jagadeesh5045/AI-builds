"""Engram-style conditional memory: n-gram lookup tables that add memory
capacity without adding per-token compute.

Trending context (17 Nov 2026): Tencent published the open weights of
Hy4 flash, a 220B-parameter MoE of which roughly 64B parameters are not
transformer weights at all but *Engram tables* - hashed n-gram embedding
lookups - while only about 12B parameters are active per token. The
Engram idea (introduced by DeepSeek in early 2026) is that a large share
of what a language model stores is static factual association, and that
kind of knowledge does not need matrix-multiply compute: it can live in
a lookup table addressed by a hash of the local token n-gram, and be
fused into the residual stream through a gate.

This demo is a small, honest, stdlib-only model of that mechanism:

  * ``EngramTable`` - multi-head hashed n-gram (n = 2, 3) embedding
    tables. Storing a fact writes its value vector into one slot per
    (n-gram order, head); reading averages the addressed slots.
  * A factual-recall experiment: 200 synthetic subject -> object facts
    are stored, then recalled by nearest-neighbour decoding. Sweeping
    the table size shows the capacity story: recall is limited by hash
    collisions, not by compute, and extra heads (independent hash
    functions) dilute collisions because each head collides differently.
  * A gate: contexts whose n-grams were never stored produce a
    near-zero memory vector and the gate stays shut, so the table only
    speaks when it actually holds something.
  * Compute accounting: per-token lookup FLOPs depend on the number of
    heads, n-gram orders and the embedding width - never on the number
    of slots. That is the whole point of the architecture: memory can
    grow by orders of magnitude at constant active compute.

Nothing here trains a language model and no claim is made about Hy4
flash's quality; vendor figures in the README are labelled as reported.
The demo isolates the one mechanism the release puts at centre stage.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def hash_ngram(tokens: tuple[str, ...], seed: int) -> int:
    """Deterministic 64-bit hash of a token n-gram under a per-head seed.

    Real Engram implementations use a multiplicative/XOR hash over token
    ids; a seeded SHA-256 prefix is used here because it is stable across
    runs and platforms, which the tests rely on.
    """
    payload = f"{seed}\x1f" + "\x1f".join(tokens)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def ngrams(context: list[str], orders: tuple[int, ...] = (2, 3)) -> list[tuple[str, ...]]:
    """The trailing n-grams of a context, one per order that fits."""
    out: list[tuple[str, ...]] = []
    for n in orders:
        if len(context) >= n:
            out.append(tuple(context[-n:]))
    return out


# ---------------------------------------------------------------------------
# Small vector helpers (lists of floats; dim is tiny in this demo)
# ---------------------------------------------------------------------------


def v_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def v_scale(a: list[float], s: float) -> list[float]:
    return [x * s for x in a]


def v_norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = v_norm(a), v_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def random_unit_vector(dim: int, rng: random.Random) -> list[float]:
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    return v_scale(vec, 1.0 / v_norm(vec))


# ---------------------------------------------------------------------------
# The Engram table
# ---------------------------------------------------------------------------


@dataclass
class EngramTable:
    """Hashed n-gram embedding tables, one per head.

    ``slots`` is the memory capacity. ``heads`` independent hash
    functions address the tables; ``orders`` are the n-gram sizes read
    from the trailing context. A slot stores the *sum* of the value
    vectors written to it plus a write count, so a read returns the mean
    of whatever collided into that slot - collisions are therefore
    visible as averaged, corrupted memories, exactly the failure mode a
    capacity sweep should expose.
    """

    slots: int
    dim: int
    heads: int = 4
    orders: tuple[int, ...] = (2, 3)
    _sums: dict[tuple[int, int], list[float]] = field(init=False, repr=False)
    _counts: dict[tuple[int, int], int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.slots <= 0 or self.dim <= 0 or self.heads <= 0:
            raise ValueError("slots, dim and heads must be positive")
        # Sparse storage: only written slots exist. A real Engram table
        # is dense on disk, but the addressed set is what matters here,
        # and sparsity is what lets this demo sweep slot counts up to
        # the millions on a laptop.
        self._sums = {}
        self._counts = {}

    def _slot(self, head: int, gram: tuple[str, ...]) -> int:
        return hash_ngram(gram, seed=1_000_003 * (head + 1)) % self.slots

    def store(self, context: list[str], value: list[float]) -> int:
        """Write ``value`` under every trailing n-gram of ``context``.

        Returns the number of (order, head) slots written.
        """
        if len(value) != self.dim:
            raise ValueError("value width must match table dim")
        writes = 0
        for gram in ngrams(context, self.orders):
            for head in range(self.heads):
                key = (head, self._slot(head, gram))
                if key in self._sums:
                    self._sums[key] = v_add(self._sums[key], value)
                    self._counts[key] += 1
                else:
                    self._sums[key] = list(value)
                    self._counts[key] = 1
                writes += 1
        return writes

    def lookup(self, context: list[str]) -> tuple[list[float], int]:
        """Read the memory vector for ``context``.

        Returns (mean vector over all addressed non-empty slots, number
        of non-empty slots hit). An unseen context returns a zero vector
        and zero hits.
        """
        total = [0.0] * self.dim
        hits = 0
        for gram in ngrams(context, self.orders):
            for head in range(self.heads):
                key = (head, self._slot(head, gram))
                count = self._counts.get(key, 0)
                if count:
                    total = v_add(total, v_scale(self._sums[key], 1.0 / count))
                    hits += 1
        if hits:
            total = v_scale(total, 1.0 / hits)
        return total, hits

    def occupied_slots(self) -> int:
        """Distinct slots written in head 0 (a capacity-usage proxy)."""
        return sum(1 for (head, _slot) in self._counts if head == 0)


# ---------------------------------------------------------------------------
# Gating and fusion
# ---------------------------------------------------------------------------


def memory_gate(memory_vector: list[float]) -> float:
    """How far the memory branch may speak, in [0, 1].

    A stored fact reads back with norm close to the stored vectors'
    norm (here ~1 for a clean slot, less when collisions averaged
    unrelated facts together); an unseen context reads back the zero
    vector. The gate is a logistic on the norm, so empty or heavily
    collided memories are automatically down-weighted.
    """
    return 1.0 / (1.0 + math.exp(-(6.0 * v_norm(memory_vector) - 3.0)))


def fuse(hidden: list[float], memory_vector: list[float]) -> list[float]:
    """Residual-stream fusion: hidden + gate * memory."""
    gate = memory_gate(memory_vector)
    return v_add(hidden, v_scale(memory_vector, gate)), gate


# ---------------------------------------------------------------------------
# Synthetic factual-recall experiment
# ---------------------------------------------------------------------------


@dataclass
class FactSet:
    """``facts``: (context tokens, object id); ``objects``: id -> vector."""

    facts: list[tuple[list[str], int]]
    objects: dict[int, list[float]]


def make_facts(n_facts: int = 200, dim: int = 32, seed: int = 7) -> FactSet:
    """Synthetic 'capital of X' style associations.

    Each fact's context ends in a unique subject bigram, and each object
    is a random unit vector - the hardest case for a lookup memory,
    because objects share no structure the table could exploit.
    """
    rng = random.Random(seed)
    objects = {i: random_unit_vector(dim, rng) for i in range(n_facts)}
    facts = []
    for i in range(n_facts):
        context = ["the", f"subj{i:03d}", "capital"]
        facts.append((context, i))
    return FactSet(facts=facts, objects=objects)


def recall_at_1(table: EngramTable, fact_set: FactSet) -> float:
    """Fraction of facts whose memory vector decodes to the right object."""
    correct = 0
    for context, object_id in fact_set.facts:
        memory, hits = table.lookup(context)
        if hits == 0:
            continue
        best_id = max(
            fact_set.objects, key=lambda oid: cosine(memory, fact_set.objects[oid])
        )
        if best_id == object_id:
            correct += 1
    return correct / len(fact_set.facts)


def capacity_sweep(
    fact_set: FactSet, slot_counts: tuple[int, ...], head_counts: tuple[int, ...]
) -> list[dict]:
    """Recall@1 as a function of table slots and head count."""
    rows = []
    for heads in head_counts:
        for slots in slot_counts:
            table = EngramTable(slots=slots, dim=len(next(iter(fact_set.objects.values()))), heads=heads)
            for context, object_id in fact_set.facts:
                table.store(context, fact_set.objects[object_id])
            rows.append(
                {
                    "heads": heads,
                    "slots": slots,
                    "recall_at_1": recall_at_1(table, fact_set),
                    "occupied_head0": table.occupied_slots(),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Compute and memory accounting
# ---------------------------------------------------------------------------


def lookup_flops_per_token(heads: int, orders: int, dim: int) -> int:
    """Multiply-adds to read and fuse the memory for one token.

    Counts one fused multiply-add per dimension for averaging each
    addressed slot and one for the gated fusion. Crucially this is a
    function of heads, orders and dim only - the slot count, i.e. the
    memory capacity, does not appear.
    """
    addressed = heads * orders
    return addressed * dim + dim


def table_bytes(slots: int, heads: int, dim: int, bytes_per_value: int = 2) -> int:
    """Storage for the tables at BF16 by default."""
    return slots * heads * dim * bytes_per_value


def hy4_flash_arithmetic() -> dict:
    """The reported Hy4 flash split, as simple arithmetic.

    Figures as reported at the 17 Nov 2026 weights release: 220B total
    parameters, about 64B of them Engram tables, about 12B active per
    token, 1,048,576-token context. These are the vendor's/reported
    numbers, not measurements made here.
    """
    total_b, engram_b, active_b = 220.0, 64.0, 12.0
    return {
        "total_b": total_b,
        "engram_b": engram_b,
        "active_b": active_b,
        "engram_share_of_total": engram_b / total_b,
        "active_fraction": active_b / total_b,
        "engram_bf16_gb": engram_b * 2.0,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("Engram-style conditional memory - Hy4 flash mechanism demo")
    print("=" * 62)
    arith = hy4_flash_arithmetic()
    print("Reported Hy4 flash split (vendor/reported figures):")
    print(f"  total params      : {arith['total_b']:.0f}B")
    print(f"  Engram tables     : {arith['engram_b']:.0f}B "
          f"({arith['engram_share_of_total']:.1%} of total, "
          f"{arith['engram_bf16_gb']:.0f} GB at BF16)")
    print(f"  active per token  : {arith['active_b']:.0f}B "
          f"({arith['active_fraction']:.1%} of total)")
    print("  context           : 1,048,576 tokens")
    print()
    fact_set = make_facts()
    print(f"Factual recall: {len(fact_set.facts)} synthetic facts, "
          f"object vectors dim {len(fact_set.objects[0])}")
    print(f"  {'heads':>5} {'slots':>7} {'recall@1':>9} {'occupied(h0)':>13}")
    rows = capacity_sweep(
        fact_set, slot_counts=(256, 1024, 4096, 16384), head_counts=(1, 4)
    )
    for row in rows:
        print(f"  {row['heads']:>5} {row['slots']:>7} "
              f"{row['recall_at_1']:>9.3f} {row['occupied_head0']:>13}")
    print()
    print("Per-token lookup cost is independent of capacity:")
    for slots in (1_000, 1_000_000, 64_000_000):
        flops = lookup_flops_per_token(heads=4, orders=2, dim=64)
        gib = table_bytes(slots, heads=4, dim=64) / 2**30
        print(f"  slots={slots:>10,}  lookup FLOPs/token={flops}  "
              f"table size={gib:,.2f} GiB")
    print()
    table = EngramTable(slots=16384, dim=32, heads=4)
    for context, object_id in fact_set.facts:
        table.store(context, fact_set.objects[object_id])
    seen_context, seen_id = fact_set.facts[0]
    memory, hits = table.lookup(seen_context)
    _, gate_seen = fuse([0.0] * 32, memory)
    unseen_memory, unseen_hits = table.lookup(["the", "subj999", "capital"])
    _, gate_unseen = fuse([0.0] * 32, unseen_memory)
    print(f"Gate: stored fact -> hits={hits}, gate={gate_seen:.3f}; "
          f"unseen subject -> hits={unseen_hits}, gate={gate_unseen:.3f}")


if __name__ == "__main__":
    main()
