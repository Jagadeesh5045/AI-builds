"""Tests for the memory-native serving toy."""

import numpy as np

from memory_native import (
    DeltaMemory,
    MixtureOfStorage,
    benchmark,
    kv_cache_bytes,
    make_facts,
    memory_native_state_bytes,
    recall_at_1,
    run_full_kv,
    run_memnative,
    run_rag,
    storage_curve,
)


def test_kv_cache_bytes_formula():
    # 2 (K,V) * 1 layer * 1 head * 4 dim * 8 tokens * 2 bytes
    assert kv_cache_bytes(8, 1, 1, 4, bytes_per_elem=2) == 2 * 1 * 1 * 4 * 8 * 2


def test_kv_cache_grows_linearly():
    a = kv_cache_bytes(1000, 40, 16, 128)
    b = kv_cache_bytes(2000, 40, 16, 128)
    assert b == 2 * a


def test_memory_state_matches_paper_nano_figure():
    # Paper's MemT-1.0-Nano: 10 memory layers, 16 KV heads, d_k=d_v=256,
    # 10,485,760 state elements = 40 MiB fp32.
    elements = 10 * 16 * 256 * 256
    assert elements == 10_485_760
    assert memory_native_state_bytes(10, 16, 256, 256, 1, 4) == 40 * 1024 * 1024


def test_memory_state_size_invariant_to_history():
    curve = storage_curve([8192, 65536, 131072])
    natives = {row[2] for row in curve}
    assert len(natives) == 1  # same bytes at every history length


def test_delta_write_then_read_roundtrip():
    mem = DeltaMemory(d_k=16, d_v=8, seed=3)
    rng = np.random.default_rng(3)
    k = rng.standard_normal(16)
    v = rng.standard_normal(8)
    mem.write(k, v, beta=1.0)
    out = mem.read(k)
    assert np.allclose(out, v, atol=1e-6)


def test_gate_zero_freezes_state():
    mem = DeltaMemory(d_k=16, d_v=8, seed=3)
    before = mem.S.copy()
    rng = np.random.default_rng(9)
    mem.write(rng.standard_normal(16), rng.standard_normal(8), beta=0.0)
    assert np.array_equal(mem.S, before)


def test_second_write_supersedes_first():
    mem = DeltaMemory(d_k=16, d_v=8, seed=3)
    rng = np.random.default_rng(5)
    k = rng.standard_normal(16)
    k /= np.linalg.norm(k)
    v1 = rng.standard_normal(8)
    v2 = rng.standard_normal(8)
    mem.write(k, v1, beta=1.0)
    mem.write(k, v2, beta=1.0)  # same key, new value: delta removes the old
    out = mem.read(k)
    assert np.linalg.norm(out - v2) < np.linalg.norm(out - v1)


def test_mixture_of_storage_routes_deterministically():
    mos = MixtureOfStorage(n_experts=4, d_k=16, d_v=8, seed=0)
    rng = np.random.default_rng(7)
    k = rng.standard_normal(16)
    assert mos._route(k) == mos._route(k)
    assert 0 <= mos._route(k) < 4


def test_mixture_of_storage_roundtrip():
    mos = MixtureOfStorage(n_experts=4, d_k=32, d_v=16, seed=1)
    K, V = make_facts(40, 32, 16, seed=11)
    for i in range(40):
        mos.write(K[i], V[i], beta=1.0)
    reads = [mos.read(K[i]) for i in range(40)]
    assert recall_at_1(K, V, reads) > 0.9


def test_mos_capacity_scales_with_experts():
    one = MixtureOfStorage(1, 32, 16, seed=0)
    eight = MixtureOfStorage(8, 32, 16, seed=0)
    assert eight.n_bytes == 8 * one.n_bytes


def test_full_kv_recall_is_perfect():
    K, V = make_facts(64, 32, 16, seed=2)
    reads, nbytes = run_full_kv(K, V)
    assert recall_at_1(K, V, reads) == 1.0
    assert nbytes > 0


def test_benchmark_reports_all_strategies():
    res = benchmark(n_facts_list=(64, 128), d_k=32, d_v=16, seed=4)
    assert len(res) == 2
    for row in res:
        assert set(row) == {"n_facts", "full_kv", "memnative", "rag"}
        for key in ("full_kv", "memnative", "rag"):
            recall, nbytes = row[key]
            assert 0.0 <= recall <= 1.0
            assert nbytes > 0


def test_memnative_bytes_do_not_grow_with_facts():
    K1, V1 = make_facts(64, 32, 16, seed=6)
    K2, V2 = make_facts(512, 32, 16, seed=6)
    _, b1 = run_memnative(K1, V1, d_k=32, d_v=16)
    _, b2 = run_memnative(K2, V2, d_k=32, d_v=16)
    assert b1 == b2  # fixed state: the whole point of the paper


def test_rag_store_grows_with_facts():
    K1, V1 = make_facts(64, 32, 16, seed=8)
    K2, V2 = make_facts(512, 32, 16, seed=8)
    _, b1 = run_rag(K1, V1)
    _, b2 = run_rag(K2, V2)
    assert b2 > b1
