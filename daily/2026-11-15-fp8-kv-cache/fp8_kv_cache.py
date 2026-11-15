"""FP8 (E4M3) KV-cache demo, in pure Python (standard library only).

Trending context (13-15 Nov 2026): llama.cpp and its tensor library ggml
reached v1.0.0 on 13 November 2026. Alongside the new stable-API promise,
the release adds an FP8 KV cache (``-ctk f8_e4m3 -ctv f8_e4m3``): the
attention keys and values stored for every token in a conversation are
kept at one byte per element instead of the usual two bytes for FP16,
which halves the cache memory. One published explainer measured its own
test cache falling from 3,584 MiB to 1,792 MiB for a 32,768-token context
(that figure is the explainer's, not measured here).

This module implements the actual trade-off so it can be measured rather
than asserted:

* a bit-exact FP8 E4M3 codec (1 sign bit, 4 exponent bits, bias 7,
  3 mantissa bits, no infinity, one NaN encoding, max normal 448),
* a small KV cache that stores keys/values as FP8 bytes or FP16 bytes,
* an evaluation harness that, on seeded synthetic keys/values/queries,
  measures reconstruction error, cosine similarity, and how much the
  attention distribution itself drifts when the cache is FP8.

Nothing here claims FP8 is free: the point is to show *how much* accuracy
the halved memory costs on a controlled workload, and where the error
comes from (values near zero are represented surprisingly well thanks to
subnormals; large outliers are clipped at 448 and quantised coarsely).

Run ``python3 fp8_kv_cache.py`` for the measured demo table.
"""

from __future__ import annotations

import math
import random
import struct

# --------------------------------------------------------------------------
# FP8 E4M3 codec
# --------------------------------------------------------------------------

FP8_MAX = 448.0          # largest finite E4M3 value (0x7E)
_NAN_CODE = 0x7F         # sign 0, exponent 1111, mantissa 111


def decode_fp8_e4m3(code: int) -> float:
    """Decode one E4M3 byte (0-255) to a Python float."""
    if not 0 <= code <= 255:
        raise ValueError("FP8 code must be a byte")
    sign = -1.0 if code & 0x80 else 1.0
    exp = (code >> 3) & 0x0F
    mant = code & 0x07
    if exp == 0x0F and mant == 0x07:
        return math.nan
    if exp == 0:
        # Subnormal (or zero): value = mant/8 * 2^-6
        return sign * (mant / 8.0) * (2.0 ** -6)
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


# Precomputed decode table for every finite code, used by the encoder's
# nearest-value search. NaN codes are excluded from encoding targets.
_DECODE_TABLE = [decode_fp8_e4m3(c) for c in range(256)]
_FINITE_CODES = [c for c in range(256)
                 if not (c in (0x7F, 0xFF))]  # both NaN encodings


def encode_fp8_e4m3(value: float) -> int:
    """Encode a float to the nearest E4M3 byte (round-to-nearest, ties to
    the code with the smaller magnitude step is *not* guaranteed; ties go
    to the first, i.e. lower, code). Values beyond +/-448 are clipped to
    +/-448, matching saturating FP8 casts; NaN maps to the NaN code."""
    if math.isnan(value):
        return _NAN_CODE
    if value >= FP8_MAX:
        return 0x7E
    if value <= -FP8_MAX:
        return 0xFE
    if value == 0.0:
        return 0x80 if math.copysign(1.0, value) < 0 else 0x00
    best_code, best_err = 0, math.inf
    for code in _FINITE_CODES:
        err = abs(_DECODE_TABLE[code] - value)
        if err < best_err:
            best_code, best_err = code, err
    return best_code


def quantize_fp8(values: list[float]) -> bytes:
    return bytes(encode_fp8_e4m3(v) for v in values)


def dequantize_fp8(data: bytes) -> list[float]:
    return [_DECODE_TABLE[b] for b in data]


def quantize_f16(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}e", *values)


def dequantize_f16(data: bytes) -> list[float]:
    n = len(data) // 2
    return list(struct.unpack(f"<{n}e", data))


# --------------------------------------------------------------------------
# KV cache
# --------------------------------------------------------------------------

class KVCache:
    """Append-only key/value cache storing one precision end to end.

    ``mode`` is ``"fp8"`` (1 byte/element) or ``"f16"`` (2 bytes/element).
    Retrieval returns dequantised floats, exactly as an attention kernel
    reading a quantised cache would see them.
    """

    def __init__(self, head_dim: int, mode: str = "fp8") -> None:
        if mode not in ("fp8", "f16"):
            raise ValueError("mode must be 'fp8' or 'f16'")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.head_dim = head_dim
        self.mode = mode
        self._keys = bytearray()
        self._values = bytearray()

    def append(self, key: list[float], value: list[float]) -> None:
        if len(key) != self.head_dim or len(value) != self.head_dim:
            raise ValueError("key/value length must equal head_dim")
        if self.mode == "fp8":
            self._keys += quantize_fp8(key)
            self._values += quantize_fp8(value)
        else:
            self._keys += quantize_f16(key)
            self._values += quantize_f16(value)

    def __len__(self) -> int:
        bytes_per_token = self.head_dim * (1 if self.mode == "fp8" else 2)
        return len(self._keys) // bytes_per_token

    @property
    def memory_bytes(self) -> int:
        return len(self._keys) + len(self._values)

    def _slice(self, store: bytearray, index: int) -> list[float]:
        width = self.head_dim * (1 if self.mode == "fp8" else 2)
        chunk = bytes(store[index * width:(index + 1) * width])
        if len(chunk) != width:
            raise IndexError(index)
        return dequantize_fp8(chunk) if self.mode == "fp8" \
            else dequantize_f16(chunk)

    def key(self, index: int) -> list[float]:
        return self._slice(self._keys, index)

    def value(self, index: int) -> list[float]:
        return self._slice(self._values, index)


def cache_memory_bytes(n_tokens: int, n_layers: int, n_kv_heads: int,
                       head_dim: int, bytes_per_element: int) -> int:
    """Total KV-cache bytes: keys AND values, every layer, every token."""
    return (2 * n_layers * n_tokens * n_kv_heads * head_dim
            * bytes_per_element)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def evaluate(seed: int = 20261115, n_tokens: int = 512,
             head_dim: int = 64, n_queries: int = 32) -> dict:
    """Measure FP8 vs FP16 cache fidelity on seeded synthetic data.

    Keys/values are standard-normal (a reasonable first model of real
    activations before outlier channels are considered). Returns, per
    mode: relative RMSE of the reconstructed keys, mean cosine
    similarity, attention top-1 agreement with the exact (float) cache,
    mean KL divergence of the attention distribution, and bytes stored.
    """
    rng = random.Random(seed)
    keys = [[rng.gauss(0.0, 1.0) for _ in range(head_dim)]
            for _ in range(n_tokens)]
    values = [[rng.gauss(0.0, 1.0) for _ in range(head_dim)]
              for _ in range(n_tokens)]
    queries = [[rng.gauss(0.0, 1.0) for _ in range(head_dim)]
               for _ in range(n_queries)]

    exact_probs = []
    for q in queries:
        scores = [sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(head_dim)
                  for k in keys]
        exact_probs.append(_softmax(scores))

    results: dict[str, dict] = {}
    for mode in ("fp8", "f16"):
        cache = KVCache(head_dim, mode)
        for k, v in zip(keys, values):
            cache.append(k, v)
        sq_err = 0.0
        cosines = []
        for i, k in enumerate(keys):
            rec = cache.key(i)
            sq_err += sum((a - b) ** 2 for a, b in zip(k, rec))
            cosines.append(_cosine(k, rec))
        signal = sum(x * x for k in keys for x in k)
        top1 = 0
        kl_total = 0.0
        for qi, q in enumerate(queries):
            scores = [sum(a * b for a, b in zip(q, cache.key(i)))
                      / math.sqrt(head_dim) for i in range(n_tokens)]
            probs = _softmax(scores)
            if probs.index(max(probs)) == \
                    exact_probs[qi].index(max(exact_probs[qi])):
                top1 += 1
            kl_total += sum(p * math.log(p / max(r, 1e-12))
                            for p, r in zip(exact_probs[qi], probs)
                            if p > 0)
        results[mode] = {
            "relative_rmse": math.sqrt(sq_err / signal),
            "mean_cosine": sum(cosines) / len(cosines),
            "top1_agreement": top1 / n_queries,
            "mean_kl": kl_total / n_queries,
            "memory_bytes": cache.memory_bytes,
        }
    return results


def run_demo() -> dict:
    results = evaluate()
    print("FP8 (E4M3) KV cache vs FP16 cache "
          "(seeded synthetic data: 512 tokens, head_dim 64, 32 queries)")
    print(f"{'mode':<6}{'bytes':>9}{'rel.RMSE':>10}{'cosine':>9}"
          f"{'top-1':>8}{'mean KL':>10}")
    for mode in ("f16", "fp8"):
        r = results[mode]
        print(f"{mode:<6}{r['memory_bytes']:>9}{r['relative_rmse']:>10.4f}"
              f"{r['mean_cosine']:>9.4f}{r['top1_agreement']:>8.2f}"
              f"{r['mean_kl']:>10.5f}")
    ctx = 32768
    f16 = cache_memory_bytes(ctx, 32, 8, 128, 2)
    fp8 = cache_memory_bytes(ctx, 32, 8, 128, 1)
    print(f"\nCache for a {ctx:,}-token context "
          "(32 layers, 8 KV heads, head_dim 128):")
    print(f"  FP16: {f16 / 2**20:,.0f} MiB   FP8: {fp8 / 2**20:,.0f} MiB"
          f"   saving: {100 * (1 - fp8 / f16):.0f}%")
    print("\nReading: FP8 halves the cache. On well-behaved (roughly "
          "normal) activations\nthe attention distribution barely moves; "
          "the risk is outlier channels,\nwhose large values are quantised "
          "coarsely and clipped at 448, which is why\nproduction FP8 caches "
          "pair the format with per-token or per-channel scaling.")
    return results


if __name__ == "__main__":
    run_demo()
