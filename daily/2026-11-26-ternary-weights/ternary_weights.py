"""
Ternary weight quantisation in miniature.

Trending, 26 Nov 2026: PrismML released Ternary Bonsai 2.5 27B, a language
model whose every weight is constrained to one of three values, {-1, 0, +1},
cutting a reported 54 GB FP16 footprint to 5.9 GB. This module implements
the classic machinery behind that idea, from scratch in NumPy:

  * Ternary Weight Network quantisation (Li & Liu, 2016): per-row threshold
    Delta = 0.7 * mean(|w|), scale alpha = mean(|w| over the weights above
    Delta), codes in {-1, 0, +1}.
  * Bit packing: five ternary codes fit in one byte because 3**5 = 243 < 256,
    so the information-theoretic 1.585 bits per weight (log2(3)) becomes a
    practical 1.6 bits per weight on disk.
  * Multiplication-free matvec: with codes in {-1, 0, +1}, a dot product is
    only additions and subtractions of the activations, times one scale.
  * An honest cost ledger: sparsity, reconstruction error, and the size of a
    27B model under FP16, naive 2-bit codes, and 5-per-byte packing.

No model weights are downloaded and no claim is made about Bonsai's own
quality. The experiment measures what ternarisation does to a small trained
linear model built here, and reports both the compression and the error.
"""
from __future__ import annotations

import numpy as np

CODES_PER_BYTE = 5  # 3**5 = 243 <= 255
POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


# ---------------------------------------------------------------- quantise
def ternary_quantize(w: np.ndarray, threshold_factor: float = 0.7):
    """Quantise each row of ``w`` to (codes, scales).

    codes  : int8 array, same shape as w, values in {-1, 0, +1}
    scales : float array, one positive scale per row
    A weight becomes 0 when |w| <= Delta, else sign(w), with
    Delta = threshold_factor * mean(|w|) for that row. The scale is the
    mean magnitude of the weights that survived the threshold, which is
    the L2-optimal single scale for a fixed code assignment.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 1:
        w = w[None, :]
        squeeze = True
    else:
        squeeze = False
    mean_abs = np.mean(np.abs(w), axis=1, keepdims=True)
    delta = threshold_factor * mean_abs
    keep = np.abs(w) > delta
    codes = np.where(keep, np.sign(w), 0).astype(np.int8)
    kept_mag = np.where(keep, np.abs(w), 0.0)
    counts = np.maximum(keep.sum(axis=1, keepdims=True), 1)
    scales = (kept_mag.sum(axis=1, keepdims=True) / counts).ravel()
    codes2d = codes
    if squeeze:
        return codes2d[0], float(scales[0])
    return codes2d, scales


def dequantize(codes: np.ndarray, scales) -> np.ndarray:
    """Rebuild approximate weights: scale[row] * codes[row]."""
    codes = np.asarray(codes)
    scales = np.asarray(scales, dtype=np.float64)
    if codes.ndim == 1:
        return codes.astype(np.float64) * float(scales)
    return codes.astype(np.float64) * scales[:, None]


# ----------------------------------------------------------------- packing
def pack_codes(codes: np.ndarray) -> np.ndarray:
    """Pack ternary codes 5 per byte. Code c in {-1,0,+1} -> digit c+1 in
    {0,1,2}; a byte holds digits d0..d4 as sum(d_i * 3**i) <= 242."""
    flat = (np.asarray(codes).ravel().astype(np.int64) + 1)
    pad = (-len(flat)) % CODES_PER_BYTE
    if pad:
        flat = np.concatenate([flat, np.ones(pad, dtype=np.int64)])  # digit 1 == code 0
    grouped = flat.reshape(-1, CODES_PER_BYTE)
    return (grouped * POW3).sum(axis=1).astype(np.uint8)


def unpack_codes(packed: np.ndarray, count: int) -> np.ndarray:
    """Inverse of pack_codes, returning exactly ``count`` codes."""
    vals = np.asarray(packed, dtype=np.uint16)[:, None]
    digits = (vals // POW3[None, :]) % 3
    return (digits.ravel()[:count].astype(np.int8) - 1)


# ------------------------------------------------------- add-only inference
def ternary_matvec(codes: np.ndarray, scales, x: np.ndarray) -> np.ndarray:
    """y = (scales * codes) @ x computed without any weight multiplies.

    For each row, y_i / scale_i = sum(x_j where code=+1) - sum(x_j where
    code=-1). Zero codes contribute nothing, so the work also shrinks with
    sparsity. The only multiplications are one scale per output row.
    """
    codes = np.asarray(codes)
    x = np.asarray(x, dtype=np.float64)
    if codes.ndim == 1:
        pos = x[codes == 1].sum()
        neg = x[codes == -1].sum()
        return np.array([float(scales) * (pos - neg)])
    out = np.empty(codes.shape[0], dtype=np.float64)
    for i, row in enumerate(codes):
        out[i] = x[row == 1].sum() - x[row == -1].sum()
    return out * np.asarray(scales, dtype=np.float64)


# -------------------------------------------------------------------- size
def size_report(n_params: int) -> dict:
    """Bytes for one model under three encodings (scales excluded: one
    float per row is negligible next to billions of weights)."""
    fp16 = n_params * 2
    two_bit = int(np.ceil(n_params * 2 / 8))
    packed = int(np.ceil(n_params / CODES_PER_BYTE))
    return {
        "params": n_params,
        "fp16_bytes": fp16,
        "two_bit_bytes": two_bit,
        "packed_bytes": packed,
        "bits_per_weight_theory": float(np.log2(3)),
        "packed_ratio_vs_fp16": fp16 / packed,
    }


def sparsity(codes: np.ndarray) -> float:
    return float(np.mean(np.asarray(codes) == 0))


# -------------------------------------------------------------------- demo
def main() -> None:
    rng = np.random.default_rng(7)

    # A small trained-by-construction problem: a teacher linear layer whose
    # outputs we ternarise, so error is measured against known weights.
    w = rng.normal(0, 1, size=(64, 128))
    # Make the weight distribution heavier in the middle, as trained nets are.
    w = w * rng.uniform(0.3, 1.0, size=(64, 1))
    codes, scales = ternary_quantize(w)
    w_hat = dequantize(codes, scales)
    rel_err = float(np.linalg.norm(w - w_hat) / np.linalg.norm(w))
    print(f"layer shape            : {w.shape}")
    print(f"sparsity (zero codes)  : {sparsity(codes):.3f}")
    print(f"relative weight error  : {rel_err:.4f}")

    x = rng.normal(0, 1, size=128)
    y_ref = w @ x
    y_ter = ternary_matvec(codes, scales, x)
    cos = float(y_ref @ y_ter / (np.linalg.norm(y_ref) * np.linalg.norm(y_ter)))
    print(f"matvec cosine vs FP    : {cos:.4f}")
    print(f"packed roundtrip exact : {np.array_equal(unpack_codes(pack_codes(codes), codes.size), codes.ravel())}")

    rep = size_report(27_000_000_000)
    print(f"27B FP16 size          : {rep['fp16_bytes']/1e9:.1f} GB")
    print(f"27B packed ternary     : {rep['packed_bytes']/1e9:.1f} GB "
          f"({rep['packed_ratio_vs_fp16']:.2f}x smaller, "
          f"{rep['bits_per_weight_theory']:.3f} bits/weight in theory)")
    print("Reported for Bonsai 2.5 27B: 54 GB -> 5.9 GB. The packed figure")
    print("here is 5.4 GB; the gap is scales, metadata and non-weight data.")


if __name__ == "__main__":
    main()
