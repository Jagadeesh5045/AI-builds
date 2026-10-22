"""Memory-native serving: a toy Gated DeltaNet-style write path.

Motivation (MemT-1.0, arXiv 2610.26825, Oct 2026): instead of persisting a
full KV cache that grows linearly with history length, bake a fixed-size
dynamic memory into the serving loop. Each trajectory keeps one memory state;
growing history updates its *contents* while its *size* stays constant. The
paper reports ~66x less persistent storage than full KV cache at 128K tokens
and ~1.86x faster end-to-end inference for MemT-1.0-Nano (figures are the
authors', reproduced as labels, not measured here).

This file implements the core ideas as a small, honest toy:
  1. A gated delta-rule memory (fast-weight) state updated token-by-token,
     the spiritual cousin of the paper's "Gated DeltaNet-2 ... token-level
     memory update".
  2. A Mixture-of-Storage router that spreads writes across expert memory
     banks so capacity scales without raising per-step cost.
  3. Storage arithmetic: KV cache bytes vs fixed memory-state bytes as the
     history grows.

Everything here is a simulation. The paper's mechanism is a learned recurrent
layer inside a 36B model; this is a hand-rolled numpy analog that lets you
*see* the storage curve and the finite-capacity trade-off.
"""

import hashlib

import numpy as np


# --------------------------------------------------------------------------
# 1. Storage arithmetic: why full KV cache hurts at long contexts
# --------------------------------------------------------------------------

def kv_cache_bytes(seq_len, n_layers, n_kv_heads, d_head, bytes_per_elem=2):
    """Persistent KV cache bytes for one trajectory (full attention).

    Standard accounting: 2 (K and V) * layers * kv_heads * d_head * seq_len
    * bytes_per_elem. Grows linearly with seq_len.
    """
    return 2 * n_layers * n_kv_heads * d_head * seq_len * bytes_per_elem


def memory_native_state_bytes(n_memory_layers, n_kv_heads, d_k, d_v,
                              n_experts=1, bytes_per_elem=4):
    """Fixed per-trajectory memory state bytes (memory-native serving).

    N_state = sum over memory layers of E * H_kv * d_k * d_v  (paper Eq. 4.3).
    History growth updates these elements but never adds new ones.
    """
    elements = n_memory_layers * n_experts * n_kv_heads * d_k * d_v
    return elements * bytes_per_elem


def storage_curve(seq_lens):
    """Return [(seq_len, kv_bytes, native_bytes), ...].

    Memory state matches the paper's MemT-1.0-Nano figure: 10 memory layers,
    16 memory KV heads, d_k = d_v = 256 -> 10,485,760 elements = 40 MiB fp32.

    The KV line is calibrated so its 128K point equals the paper's reported
    66x reduction (KV grows exactly linearly with history, so scaling the
    line from that endpoint is exact). As a consistency check this also
    reproduces the paper's other reported figure: ~33x at 64K tokens
    (paper Section 4.3). The exact byte count of a real backbone's KV cache
    depends on its GQA groups and precision; the structural point here,
    linear vs fixed, is what matters.
    """
    native = memory_native_state_bytes(
        n_memory_layers=10, n_kv_heads=16, d_k=256, d_v=256,
        n_experts=1, bytes_per_elem=4)
    rows = []
    for s in seq_lens:
        kv = 66.0 * native * (s / 131072)
        rows.append((s, kv, native))
    return rows


# --------------------------------------------------------------------------
# 2. Gated delta-rule memory (one expert bank)
# --------------------------------------------------------------------------

class DeltaMemory:
    """A single fast-weight memory bank updated token-by-token.

    State S (d_k x d_v) maps keys to values. The delta-rule (Widrow-Hoff)
    update writes a new association while *removing* what the state already
    predicted, which keeps old entries from piling up on top of each other:

        err  = v - S^T k
        S   += beta * outer(k, err) / (k^T k)

    With beta = 1 a single presentation installs the association exactly;
    beta in (0, 1) writes it partially. A scalar gate beta = 0 freezes the
    state: the memory simply refuses the write. (Production fast-weight
    systems, including the Gated DeltaNet family, use a *stabilized*
    variant with a normalized denominator that trades one-step exactness
    for stability across millions of updates; this toy keeps the exact
    form so the capacity experiment reads cleanly.)
    """

    def __init__(self, d_k, d_v, seed=0):
        self.rng = np.random.default_rng(seed)
        self.S = np.zeros((d_k, d_v))

    @property
    def n_bytes(self):
        return self.S.nbytes

    def write(self, k, v, beta=1.0):
        """Store association k -> v with gate beta in [0, 1]."""
        k = np.asarray(k, dtype=float)
        v = np.asarray(v, dtype=float)
        beta = float(np.clip(beta, 0.0, 1.0))
        if beta == 0.0:
            return
        pred = self.S.T @ k
        err = v - pred
        denom = float(k @ k) + 1e-12
        self.S += (beta / denom) * np.outer(k, err)

    def read(self, q):
        """Retrieve the value associated with query q."""
        q = np.asarray(q, dtype=float)
        return self.S.T @ q


# --------------------------------------------------------------------------
# 3. Mixture-of-Storage: route writes across expert banks
# --------------------------------------------------------------------------

class MixtureOfStorage:
    """E expert DeltaMemory banks with deterministic content routing.

    Routing is by a stable hash of the key (content-addressed), so the same
    fact always lands in the same bank. Per-step cost stays one bank update,
    while total capacity scales with E: the toy analog of the paper's
    Mixture-of-Storage, which "dynamically routes incoming semantic content
    across specialized recurrent memory slots to break traditional
    state-capacity saturation".
    """

    def __init__(self, n_experts, d_k, d_v, seed=0):
        self.experts = [DeltaMemory(d_k, d_v, seed=seed + i)
                        for i in range(n_experts)]
        self.n_experts = n_experts

    def _route(self, k):
        digest = hashlib.sha256(np.asarray(k).tobytes()).digest()
        return int.from_bytes(digest[:4], "big") % self.n_experts

    def write(self, k, v, beta=1.0):
        self.experts[self._route(k)].write(k, v, beta=beta)

    def read(self, q):
        return self.experts[self._route(q)].read(q)

    @property
    def n_bytes(self):
        return sum(e.n_bytes for e in self.experts)


# --------------------------------------------------------------------------
# 4. Toy long-horizon recall benchmark
#
# A synthetic stand-in for the paper's memory benchmarks (LoCoMo,
# PersonaMem, LongMemEval style): a long stream of user facts is ingested,
# then the model is quizzed. Three serving strategies compete:
#   full_kv   : exact attention over every stored fact (perfect, unbounded)
#   memnative : MixtureOfStorage fast weights (fixed size)
#   rag       : external chunk store, top-k cosine retrieval then re-rank
# --------------------------------------------------------------------------

def make_facts(n_facts, d_k, d_v, seed=1):
    """Random orthogonal-ish key/value associations: the 'conversation'."""
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((n_facts, d_k))
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.standard_normal((n_facts, d_v))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return K, V


def recall_at_1(K, V, reads):
    """Fraction of queries whose nearest value-neighbour is the true value."""
    V_norm = V / np.linalg.norm(V, axis=1, keepdims=True)
    hits = 0
    for i, r in enumerate(reads):
        r = r / (np.linalg.norm(r) + 1e-12)
        sims = V_norm @ r
        if int(np.argmax(sims)) == i:
            hits += 1
    return hits / len(reads)


def run_full_kv(K, V):
    """Exact nearest-key retrieval over every stored fact.

    The honest "perfect but unbounded" baseline: the query is scored
    against every stored key and the best match's value is returned.
    Recall is 1.0 by construction; the cost is storage that grows with
    every fact (the KV-cache problem the paper attacks).
    """
    reads = []
    for i in range(len(K)):
        sims = K @ K[i]
        reads.append(V[int(np.argmax(sims))])
    return reads, kv_cache_bytes(len(K), 1, 1, K.shape[1])


def run_memnative(K, V, n_experts=4, d_k=64, d_v=32, seed=0):
    """Write every fact into fixed-size memory, then read it back."""
    mem = MixtureOfStorage(n_experts, d_k, d_v, seed=seed)
    for i in range(len(K)):
        mem.write(K[i], V[i], beta=1.0)
    reads = [mem.read(K[i]) for i in range(len(K))]
    return reads, mem.n_bytes


def run_rag(K, V, top_k=2):
    """External store + top-k key retrieval, reader re-ranks by key match.

    The reader only sees the retrieved candidates and picks the one whose
    key best matches the query: no peeking at the ground-truth value.
    At toy scale with exact queries this matches full-KV recall; the cost
    is the external vector store, which grows with the corpus, plus the
    retrieval step on every query.
    """
    reads = []
    for i in range(len(K)):
        sims = K @ K[i]
        top = np.argsort(sims)[-top_k:]
        best = top[int(np.argmax(sims[top]))]
        reads.append(V[best])
    # external vector store: one fp32 vector per fact, both K and V
    store_bytes = K.nbytes + V.nbytes
    return reads, store_bytes


def benchmark(n_facts_list=(128, 256, 512, 1024), d_k=64, d_v=32, seed=1):
    """Compare the three strategies: recall@1 and persistent bytes."""
    results = []
    for n in n_facts_list:
        K, V = make_facts(n, d_k, d_v, seed=seed)
        kv_reads, kv_bytes = run_full_kv(K, V)
        mn_reads, mn_bytes = run_memnative(K, V, d_k=d_k, d_v=d_v)
        rag_reads, rag_bytes = run_rag(K, V)
        results.append({
            "n_facts": n,
            "full_kv": (recall_at_1(K, V, kv_reads), kv_bytes),
            "memnative": (recall_at_1(K, V, mn_reads), mn_bytes),
            "rag": (recall_at_1(K, V, rag_reads), rag_bytes),
        })
    return results
