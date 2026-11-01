"""
gguf_quant.py - Why a 35B model can run on a CPU: GGUF block quantisation.

Trending context (1 Nov 2026): VIDRAFT's POCKET-35B-GGUF was spotlighted
today as a 35-billion-parameter model packaged in the GGUF format to run
locally on ordinary CPU hardware with no GPU. The enabling trick is not
magic, it is arithmetic:

  * FP16 weights for 35B parameters need 35e9 * 2 bytes = 70 GB, which
    fits in no ordinary desktop's RAM alongside anything else.
  * GGUF-style block quantisation stores most weights at 4 bits each,
    in blocks of 32 weights that share one FP16 scale factor. Effective
    cost is 4 + 16/32 = 4.5 bits per weight, so the same 35B model is
    about 19.7 GB, which does fit in a 32 GB machine, and a CPU only
    has to stream those 19.7 GB from RAM per token instead of 70 GB.

This module implements that scheme (a simplified Q4_0 / Q8_0 style
symmetric per-block quantiser), measures the actual round-trip error
it introduces on a synthetic weight matrix and a matrix-vector
product, and prints the exact memory table for a 35B model.

Simplifications vs real GGUF: real Q4_K uses super-blocks with
quantised scales and mins; here each block of 32 weights carries one
FP16 scale and symmetric signed codes. The memory and error orders of
magnitude are the same, and every number printed is computed, not
asserted.

Run:  python3 gguf_quant.py
Tests: python3 -m unittest test_gguf_quant -v
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

BLOCK_SIZE = 32          # weights per block (GGUF Q4_0 / Q8_0 use 32)
SCALE_BYTES = 2          # one FP16 scale per block
MODEL_PARAMS = 35_000_000_000   # POCKET-35B scale


@dataclass
class Quantized:
    codes: np.ndarray    # int8 container; for 4-bit, values in [-8, 7]
    scales: np.ndarray   # one float64 scale per block (stored as FP16)
    shape: tuple
    bits: int
    block_size: int


def _qmax(bits: int) -> int:
    """Largest positive code for signed symmetric quantisation."""
    return 2 ** (bits - 1) - 1


def quantize(weights: np.ndarray, bits: int, block_size: int = BLOCK_SIZE) -> Quantized:
    """Symmetric per-block quantisation.

    Each block of `block_size` consecutive weights shares one scale,
    set so the largest-magnitude weight in the block maps to +/-qmax.
    A zero block gets scale 1.0 and all-zero codes (dequantises to 0).
    """
    if bits not in (4, 8):
        raise ValueError("this demo implements 4-bit and 8-bit GGUF-style formats")
    flat = np.asarray(weights, dtype=np.float64).ravel()
    pad = (-len(flat)) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    blocks = flat.reshape(-1, block_size)
    amax = np.max(np.abs(blocks), axis=1)
    qmax = _qmax(bits)
    scales = np.where(amax > 0, amax / qmax, 1.0)
    codes = np.rint(blocks / scales[:, None])
    codes = np.clip(codes, -qmax - 1, qmax).astype(np.int8)
    return Quantized(codes=codes, scales=scales, shape=np.asarray(weights).shape,
                     bits=bits, block_size=block_size)


def dequantize(q: Quantized) -> np.ndarray:
    """Reconstruct approximate weights: code * block scale, padding removed."""
    flat = (q.codes.astype(np.float64) * q.scales[:, None]).ravel()
    n = int(np.prod(q.shape))
    return flat[:n].reshape(q.shape)


def packed_code_bytes(n_weights: int, bits: int) -> int:
    """Bytes for the codes alone: 8-bit = 1/weight, 4-bit = 2/weight packed."""
    return math.ceil(n_weights * bits / 8)


def storage_bytes(n_weights: int, bits: int | None, block_size: int = BLOCK_SIZE) -> int:
    """Total bytes to store `n_weights` weights.

    bits=None means the FP16 baseline (2 bytes/weight, no scales).
    Otherwise: packed codes plus one FP16 scale per block.
    """
    if bits is None:
        return n_weights * 2
    n_blocks = math.ceil(n_weights / block_size)
    return packed_code_bytes(n_weights, bits) + n_blocks * SCALE_BYTES


def effective_bits_per_weight(bits: int, block_size: int = BLOCK_SIZE) -> float:
    """Code bits plus the amortised scale cost: bits + 16/block_size."""
    return bits + (SCALE_BYTES * 8) / block_size


def roundtrip_metrics(weights: np.ndarray, bits: int) -> dict:
    """Quantise + dequantise and measure the damage."""
    q = quantize(weights, bits)
    approx = dequantize(q)
    err = approx - weights
    denom = np.linalg.norm(weights)
    return {
        "bits": bits,
        "mse": float(np.mean(err ** 2)),
        "max_abs_error": float(np.max(np.abs(err))),
        "relative_l2_error": float(np.linalg.norm(err) / denom) if denom else 0.0,
    }


def matvec_metrics(rows: int = 256, cols: int = 256, seed: int = 7) -> list[dict]:
    """Error on y = W x, the operation a CPU actually runs per token."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((rows, cols)) / math.sqrt(cols)
    x = rng.standard_normal(cols)
    y_ref = w @ x
    out = []
    for bits in (8, 4):
        y_q = dequantize(quantize(w, bits)) @ x
        cos = float(y_ref @ y_q / (np.linalg.norm(y_ref) * np.linalg.norm(y_q)))
        rel = float(np.linalg.norm(y_q - y_ref) / np.linalg.norm(y_ref))
        out.append({"bits": bits, "output_cosine": cos, "output_rel_error": rel})
    return out


def memory_table(params: int = MODEL_PARAMS) -> list[dict]:
    rows = []
    for label, bits in (("FP16", None), ("Q8_0-style", 8), ("Q4_0-style", 4)):
        total = storage_bytes(params, bits)
        rows.append({
            "format": label,
            "bytes": total,
            "gb": total / 1e9,
            "bits_per_weight": 16.0 if bits is None else effective_bits_per_weight(bits),
            "fits_32gb_ram": total < 32e9,
        })
    return rows


def main() -> None:
    print("GGUF block quantisation: how 35B parameters fit on a CPU-only box")
    print(f"(block size {BLOCK_SIZE} weights, one FP16 scale per block)\n")

    print(f"Memory for {MODEL_PARAMS/1e9:.0f}B parameters:")
    for r in memory_table():
        fits = "fits in 32 GB RAM" if r["fits_32gb_ram"] else "does NOT fit in 32 GB RAM"
        print(f"  {r['format']:<11} {r['gb']:6.2f} GB  "
              f"({r['bits_per_weight']:.1f} bits/weight)  {fits}")
    fp16 = storage_bytes(MODEL_PARAMS, None)
    q4 = storage_bytes(MODEL_PARAMS, 4)
    print(f"  Q4_0-style is {fp16/q4:.2f}x smaller than FP16, and per token the CPU")
    print(f"  streams {q4/1e9:.1f} GB from RAM instead of {fp16/1e9:.0f} GB. On a CPU,")
    print("  generation is memory-bandwidth bound, so bytes per weight is speed.\n")

    rng = np.random.default_rng(7)
    w = rng.standard_normal((256, 256)) / 16.0
    print("Round-trip error on a 256x256 synthetic weight matrix:")
    for bits in (8, 4):
        m = roundtrip_metrics(w, bits)
        print(f"  {bits}-bit  MSE {m['mse']:.3e}  max|err| {m['max_abs_error']:.4f}  "
              f"relative L2 {m['relative_l2_error']:.4f}")
    print("Matrix-vector product y = W x (what inference computes):")
    for m in matvec_metrics():
        print(f"  {m['bits']}-bit  output cosine {m['output_cosine']:.5f}  "
              f"relative error {m['output_rel_error']:.4f}")

    big = rng.standard_normal((1024, 1024)) / 32.0
    t0 = time.perf_counter()
    _ = dequantize(quantize(big, 4)) @ rng.standard_normal(1024)
    dt = time.perf_counter() - t0
    print(f"\nToy timing on this machine: dequantise + matvec for a 1024x1024 "
          f"layer took {dt*1000:.1f} ms (illustrative only, not a benchmark).")


if __name__ == "__main__":
    main()
