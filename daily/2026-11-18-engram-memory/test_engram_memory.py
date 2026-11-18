"""Tests for the Engram-style conditional memory demo.

Run with: python3 -m pytest test_engram_memory.py
or plain:  python3 test_engram_memory.py
"""

import math

from engram_memory import (
    EngramTable,
    capacity_sweep,
    cosine,
    fuse,
    hash_ngram,
    hy4_flash_arithmetic,
    lookup_flops_per_token,
    make_facts,
    memory_gate,
    ngrams,
    recall_at_1,
    table_bytes,
    v_norm,
)


def test_hash_is_deterministic_and_seed_sensitive():
    gram = ("subj042", "capital")
    assert hash_ngram(gram, seed=7) == hash_ngram(gram, seed=7)
    assert hash_ngram(gram, seed=7) != hash_ngram(gram, seed=8)
    assert hash_ngram(gram, seed=7) != hash_ngram(("subj043", "capital"), seed=7)


def test_ngrams_take_trailing_context():
    grams = ngrams(["a", "b", "c", "d"], orders=(2, 3))
    assert grams == [("c", "d"), ("b", "c", "d")]
    assert ngrams(["only"], orders=(2, 3)) == []


def test_store_and_lookup_roundtrip_without_collisions():
    fact_set = make_facts(n_facts=20, dim=16, seed=3)
    table = EngramTable(slots=1_000_000, dim=16, heads=2)
    for context, object_id in fact_set.facts:
        table.store(context, fact_set.objects[object_id])
    context, object_id = fact_set.facts[5]
    memory, hits = table.lookup(context)
    assert hits == 4  # 2 orders x 2 heads
    assert cosine(memory, fact_set.objects[object_id]) > 0.999


def test_unseen_context_reads_back_empty():
    table = EngramTable(slots=1024, dim=8, heads=2)
    memory, hits = table.lookup(["the", "subj999", "capital"])
    assert hits == 0
    assert v_norm(memory) == 0.0


def test_recall_is_perfect_with_a_large_table():
    fact_set = make_facts(n_facts=200, dim=32, seed=7)
    table = EngramTable(slots=1_000_000, dim=32, heads=4)
    for context, object_id in fact_set.facts:
        table.store(context, fact_set.objects[object_id])
    assert recall_at_1(table, fact_set) == 1.0


def test_collisions_damage_recall_in_a_small_table():
    fact_set = make_facts(n_facts=200, dim=32, seed=7)
    table = EngramTable(slots=64, dim=32, heads=1)
    for context, object_id in fact_set.facts:
        table.store(context, fact_set.objects[object_id])
    # 200 facts into 64 slots guarantees heavy collision; recall must
    # fall well below the collision-free result.
    assert recall_at_1(table, fact_set) < 0.6


def test_more_slots_never_hurt_recall_in_sweep():
    fact_set = make_facts(n_facts=200, dim=32, seed=7)
    rows = capacity_sweep(fact_set, slot_counts=(256, 16384), head_counts=(1,))
    by_slots = {row["slots"]: row["recall_at_1"] for row in rows}
    assert by_slots[16384] >= by_slots[256]


def test_extra_heads_help_at_small_capacity():
    fact_set = make_facts(n_facts=200, dim=32, seed=7)
    rows = capacity_sweep(fact_set, slot_counts=(256,), head_counts=(1, 4))
    by_heads = {row["heads"]: row["recall_at_1"] for row in rows}
    # Independent hash functions collide on different facts, so the
    # averaged multi-head read keeps more of each fact intact.
    assert by_heads[4] > by_heads[1]


def test_gate_stays_shut_for_empty_memory_and_opens_for_stored():
    assert memory_gate([0.0] * 32) < 0.05
    fact_set = make_facts(n_facts=50, dim=32, seed=11)
    table = EngramTable(slots=1_000_000, dim=32, heads=4)
    for context, object_id in fact_set.facts:
        table.store(context, fact_set.objects[object_id])
    memory, _ = table.lookup(fact_set.facts[0][0])
    assert memory_gate(memory) > 0.8


def test_fuse_returns_gate_and_preserves_hidden_when_gate_shut():
    hidden = [1.0] + [0.0] * 31
    fused, gate = fuse(hidden, [0.0] * 32)
    assert gate < 0.05
    assert fused == hidden


def test_lookup_flops_do_not_depend_on_slots():
    # The signature has no slot argument at all; assert the accounting
    # identity the architecture rests on across table sizes instead.
    flops = lookup_flops_per_token(heads=4, orders=2, dim=64)
    assert flops == 4 * 2 * 64 + 64
    sizes = [table_bytes(s, heads=4, dim=64) for s in (1_000, 1_000_000)]
    assert sizes[1] == 1000 * sizes[0]  # memory scales, compute does not


def test_hy4_arithmetic_matches_reported_split():
    arith = hy4_flash_arithmetic()
    assert arith["total_b"] == 220.0
    assert math.isclose(arith["engram_share_of_total"], 64.0 / 220.0)
    assert math.isclose(arith["active_fraction"], 12.0 / 220.0)
    assert arith["engram_bf16_gb"] == 128.0


def test_store_rejects_wrong_width():
    table = EngramTable(slots=16, dim=8)
    try:
        table.store(["a", "b"], [0.0] * 4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched value width")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passing")
