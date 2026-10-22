# Memory-native serving: a toy of MemT-1.0's write path

**Trending topic (20 Oct 2026):** *MemT-1.0: Towards Memory-Native Evolution*
(arXiv 2610.26825). The paper argues long-context serving should stop
carrying a full KV cache and instead bake a fixed-size *dynamic memory* into
the model. Two models: **MemT-1.0-Nano** (36B params, 3B active, 10M dynamic
memory, Qwen3.6 backbone) and **MemT-1.0-Flash** (288B, 17B active, 336M
dynamic memory, DeepSeek-V4-Flash backbone). Mechanism: **Gated DeltaNet-2**
with a stable token-level memory update, plus **Mixture-of-Storage** routing
to scale capacity. Reported results (the authors' figures, not measured
here): ~66x less persistent storage than full KV cache at 128K tokens,
~1.86x inference speedup, ~33x at 64K, +7.6 pts on SWE-bench Verified from
internalised experience. Evaluated on MemBench, PersonaMem, LoCoMo,
LongMemEval-S, MemoryAgentBench, BEAM-1M.

## What this demo implements

A small, honest numpy analog of the core ideas. It is a simulation of the
*storage and capacity* argument, not a replica of the learned layer.

1. **Gated delta-rule memory** (`DeltaMemory`): a fast-weight state
   `S (d_k x d_v)` updated token-by-token with the Widrow-Hoff rule, gated
   by a scalar beta. Beta = 0 refuses the write entirely. Rewriting the same
   key supersedes the old value instead of stacking on it.
2. **Mixture-of-Storage** (`MixtureOfStorage`): E expert banks with
   deterministic content-addressed routing (SHA-256 of the key), so capacity
   scales with E while each step still costs one bank update.
3. **Storage arithmetic**: KV cache bytes grow linearly with history; the
   memory-native state is fixed at the paper's Nano figure (10,485,760 fp32
   elements = 40 MiB). The KV line is calibrated so its 128K point equals
   the paper's reported 66x, which also reproduces their 33x at 64K, since
   KV growth is exactly linear.
4. **Recall benchmark**: a synthetic long-horizon fact-recall task
   (LoCoMo/PersonaMem flavour). Full-KV nearest-key retrieval and top-2 RAG
   are perfect but their storage grows with every fact; the fixed-size
   memory holds recall until capacity saturates, then degrades gracefully.

## Run it

```bash
python3 benchmark.py            # storage curve + recall table
python3 -m pytest test_memory_native.py -q   # 14 tests (or run each test_* fn)
```

## Results (toy scale, exact queries)

| facts | full-KV recall | mem-native recall | RAG recall | mem-native bytes |
|------:|---------------:|------------------:|-----------:|:-----------------|
| 128   | 1.000 | 0.969 | 1.000 | 64.0 KiB (fixed) |
| 256   | 1.000 | 0.789 | 1.000 | 64.0 KiB (fixed) |
| 512   | 1.000 | 0.395 | 1.000 | 64.0 KiB (fixed) |
| 1024  | 1.000 | 0.172 | 1.000 | 64.0 KiB (fixed) |

The fixed state is the whole point: 64 KiB whether it holds 128 facts or
1024. The honest cost is finite capacity, and the paper's Discussion section
examines exactly this effect at real scale. Real systems close the gap with
learned updates, many more dimensions, and Mixture-of-Storage experts; this
toy uses 4 experts x 64 x 32.

## Honest proxy caveats

- The paper's memory is a *learned* recurrent layer in a 36B model trained
  on 34B tokens; this is a hand-rolled delta rule with random keys.
- The 66x/1.86x/33x figures are the authors' measurements, reproduced as
  labels and calibration points, not independently verified.
- Toy RAG matches full-KV recall here because queries are exact; real RAG
  pays retrieval latency and can miss (the fragmentation the paper cites).
