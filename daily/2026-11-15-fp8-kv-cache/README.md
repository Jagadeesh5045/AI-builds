# FP8 KV cache: what halving the attention cache actually costs

**Trending topic (13-15 Nov 2026):** llama.cpp and its tensor library ggml
reached **v1.0.0 on 13 November 2026**, their first stable release after
about three and a half years. The headline is an API-stability promise,
but the release also adds an **FP8 KV cache** (`-ctk f8_e4m3 -ctv f8_e4m3`):
the keys and values a model stores for every token are kept at **1 byte
per element instead of 2** (FP16), halving the memory of long contexts.
One explainer's own test measured its cache falling from 3,584 MiB to
1,792 MiB at a 32,768-token context (that measurement is the explainer's;
the numbers below are this demo's own, on synthetic data).

## What this code shows

Pure Python, standard library only:

- `fp8_kv_cache.py`
  - A bit-exact **FP8 E4M3 codec**: 1 sign bit, 4 exponent bits (bias 7),
    3 mantissa bits, subnormals down to 0.001953125, no infinity, one NaN
    encoding, maximum finite value 448, saturating encode.
  - A small append-only **KV cache** that stores keys/values as FP8 or
    FP16 bytes and returns dequantised vectors on read, the way an
    attention kernel consuming a quantised cache would see them.
  - An **evaluation harness**: seeded synthetic keys/values/queries,
    measuring reconstruction error and, more importantly, how much the
    *attention distribution itself* drifts versus the exact cache.
- `test_fp8_kv_cache.py` — 16 tests: known decode values, round-to-nearest
  encoding, clipping and NaN handling, the exact 2:1 memory ratio, and the
  attention-agreement bounds below.

## Measured results (this demo, seeded)

512 cached tokens, head_dim 64, 32 queries, standard-normal activations:

| mode | bytes stored | relative RMSE | mean cosine | attention top-1 agreement | mean KL vs exact |
|---|---|---|---|---|---|
| FP16 | 131,072 | 0.0002 | 1.0000 | 1.00 | 0.00000 |
| FP8 E4M3 | 65,536 | 0.0268 | 0.9997 | 0.94 | 0.00036 |

For a 32,768-token context on a 32-layer model with 8 KV heads and
head_dim 128, the cache formula in the code gives **4,096 MiB in FP16
versus 2,048 MiB in FP8**, a flat 50% saving at any scale, because the
format is exactly half the width.

## The honest caveats

- **Well-behaved activations are the easy case.** On roughly normal data
  the attention distribution barely moves (mean KL 0.00036). Real models
  have *outlier channels*: a few dimensions with values 10-100x the rest.
  Those are quantised coarsely by a shared 3-bit mantissa and anything
  past 448 is clipped outright, which is where FP8 caches lose accuracy
  in practice, and why production setups pair FP8 storage with per-token
  or per-channel scaling factors (not modelled here).
- **Top-1 agreement of 0.94 is not 1.00.** In 2 of 32 queries the single
  most-attended token changed. For retrieval-style attention that is
  usually harmless; for tasks that hinge on one exact token, it is the
  number to watch.
- E4M3 (more exponent, less mantissa) is the right variant for a cache:
  activations need range more than precision. Its sibling E5M2 trades
  the other way and is used mainly for gradients.

## Run it

```bash
python3 fp8_kv_cache.py      # prints the measured table above
python3 -m unittest -v       # 16 tests
```

## Sources

- ggml-org / Hugging Face blog, 13 Nov 2026: llama.cpp v1.0.0 and ggml
  v1.0.0 release notes (FP8 support, stable API promise).
- systemdesign.academy explainer, 14 Nov 2026: FP8 KV-cache walkthrough
  and the 3,584 -> 1,792 MiB measurement quoted above.
