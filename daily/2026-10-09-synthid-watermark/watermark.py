"""
Provenance watermarking: a working embed-and-detect demo in the style of
Google's SynthID.

Context (Oct 2026): Google opened its SynthID Detector to the public,
letting anyone check whether an image, audio clip or video carries a
SynthID watermark. SynthID works by embedding an imperceptible signal
into AI-generated content at creation time, then detecting that signal
later. This file is an educational re-implementation of the core idea,
not Google's actual algorithm (which embeds during model sampling).

Two schemes are compared:

1. LSB embedding (spatial domain): hides bits in the least significant
   bit of pseudo-randomly chosen pixels. Fast and invisible, but fragile:
   any JPEG recompression scrambles the LSB plane.

2. DCT sign embedding (transform domain): hides one bit per 8x8 block in
   the sign of a mid-frequency DCT coefficient, with a safety margin
   `delta`. This is the same family of techniques real provenance
   watermarks use, because transform-domain signs survive the
   quantization that JPEG applies.

Run the robustness demo:
    python3 watermark.py

Run the unit tests:
    python3 test_watermark.py
"""

import hashlib
import io

import numpy as np
from PIL import Image

# Detection threshold: a score at or above this means "watermark found".
DETECT_THRESHOLD = 0.75

# Mid-frequency DCT coefficient used to carry one bit per 8x8 block.
COEFF_U, COEFF_V = 3, 4


def _rng(seed: str) -> np.random.Generator:
    """Deterministic PRNG derived from a text seed (the watermark key)."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def text_to_bits(text: str) -> list:
    """Encode text as bits: 16-bit big-endian length prefix + UTF-8 payload."""
    payload = text.encode("utf-8")
    if len(payload) > 65535:
        raise ValueError("message too long")
    bits = [(len(payload) >> i) & 1 for i in range(15, -1, -1)]
    for byte in payload:
        bits.extend((byte >> i) & 1 for i in range(7, -1, -1))
    return bits


def bits_to_text(bits: list) -> str:
    """Inverse of text_to_bits."""
    length = 0
    for b in bits[:16]:
        length = (length << 1) | b
    payload_bits = bits[16:16 + 8 * length]
    raw = bytearray()
    for i in range(0, len(payload_bits), 8):
        byte = 0
        for b in payload_bits[i:i + 8]:
            byte = (byte << 1) | b
        raw.append(byte)
    return bytes(raw).decode("utf-8", errors="replace")


def make_test_image(size: int = 256) -> np.ndarray:
    """Deterministic synthetic grayscale image (no external files needed)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    img = 90 + 60 * (x / size) + 40 * np.sin(2 * np.pi * y / 48)
    yy, xx = y - size / 2, x - size / 2
    img += 45 * np.exp(-(xx ** 2 + yy ** 2) / (2 * (size / 6) ** 2))
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Scheme 1: LSB embedding (spatial domain)
# ---------------------------------------------------------------------------

def embed_lsb(image: np.ndarray, message: str, seed: str = "synthid-demo",
              reps: int = 5) -> np.ndarray:
    """Embed `message` in the LSB plane at seeded pseudo-random positions.

    Each bit is repeated `reps` times (repetition code) so a majority vote
    can correct a few flipped bits.
    """
    bits = text_to_bits(message)
    flat = image.reshape(-1).copy()
    needed = len(bits) * reps
    if needed > flat.size:
        raise ValueError(
            f"message needs {needed} pixels but image has {flat.size}")
    pos = _rng(seed).choice(flat.size, size=needed, replace=False)
    for i, bit in enumerate(bits):
        for p in pos[i * reps:(i + 1) * reps]:
            flat[p] = (flat[p] & 0xFE) | bit
    return flat.reshape(image.shape)


def extract_lsb(image: np.ndarray, n_bits: int, seed: str = "synthid-demo",
                reps: int = 5) -> list:
    """Recover bits embedded by embed_lsb via majority vote per bit."""
    flat = image.reshape(-1)
    pos = _rng(seed).choice(flat.size, size=n_bits * reps, replace=False)
    out = []
    for i in range(n_bits):
        votes = int((flat[pos[i * reps:(i + 1) * reps]] & 1).sum())
        out.append(1 if votes * 2 > reps else 0)
    return out


# ---------------------------------------------------------------------------
# Scheme 2: DCT sign embedding (transform domain)
# ---------------------------------------------------------------------------

def _dct_matrix(n: int = 8) -> np.ndarray:
    """Orthonormal DCT-II matrix (numpy only, no scipy needed)."""
    c = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            c[k, j] = np.cos(np.pi * k * (2 * j + 1) / (2 * n))
    c[0, :] /= np.sqrt(n)
    c[1:, :] *= np.sqrt(2 / n)
    return c


_DCT = _dct_matrix(8)


def _block_dct(block: np.ndarray) -> np.ndarray:
    return _DCT @ (block - 128.0) @ _DCT.T


def _block_idct(coeffs: np.ndarray) -> np.ndarray:
    return _DCT.T @ coeffs @ _DCT + 128.0


def embed_dct(image: np.ndarray, message: str, seed: str = "synthid-demo",
              delta: float = 30.0) -> np.ndarray:
    """Embed one bit per 8x8 block in the sign of a mid-frequency DCT coeff.

    `delta` is the safety margin: the coefficient is forced to +delta or
    -delta, so small perturbations (JPEG quantization, light noise) cannot
    flip its sign.
    """
    bits = text_to_bits(message)
    h, w = image.shape
    if h % 8 or w % 8:
        raise ValueError("image dimensions must be multiples of 8")
    n_blocks = (h // 8) * (w // 8)
    if len(bits) > n_blocks:
        raise ValueError(
            f"message needs {len(bits)} blocks but image has {n_blocks}")
    order = _rng(seed).permutation(n_blocks)
    out = image.astype(np.float64)
    for i, bit in enumerate(bits):
        b = int(order[i])
        by, bx = divmod(b, w // 8)
        block = out[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8]
        coeffs = _block_dct(block)
        coeffs[COEFF_U, COEFF_V] = delta if bit else -delta
        out[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8] = _block_idct(coeffs)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def extract_dct(image: np.ndarray, n_bits: int,
                seed: str = "synthid-demo") -> list:
    """Recover bits embedded by embed_dct by reading coefficient signs."""
    h, w = image.shape
    n_blocks = (h // 8) * (w // 8)
    order = _rng(seed).permutation(n_blocks)
    img = image.astype(np.float64)
    out = []
    for i in range(n_bits):
        b = int(order[i])
        by, bx = divmod(b, w // 8)
        block = img[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8]
        out.append(1 if _block_dct(block)[COEFF_U, COEFF_V] > 0 else 0)
    return out


# ---------------------------------------------------------------------------
# Detection: score a candidate watermark like a provenance detector would
# ---------------------------------------------------------------------------

def detect(image: np.ndarray, message: str, scheme: str = "dct",
           seed: str = "synthid-demo", **kwargs) -> tuple:
    """Return (score, detected) for `message` under the chosen scheme.

    score = fraction of extracted bits matching the message.
    A random/wrong key scores ~0.50; a real watermark scores near 1.00.
    """
    bits = text_to_bits(message)
    if scheme == "lsb":
        got = extract_lsb(image, len(bits), seed, **kwargs)
    elif scheme == "dct":
        got = extract_dct(image, len(bits), seed)
    else:
        raise ValueError("scheme must be 'lsb' or 'dct'")
    score = sum(a == b for a, b in zip(bits, got)) / len(bits)
    return score, score >= DETECT_THRESHOLD


# ---------------------------------------------------------------------------
# Robustness demo: the transforms a watermark meets in the wild
# ---------------------------------------------------------------------------

def jpeg_recompress(image: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"))


def add_noise(image: np.ndarray, sigma: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float64) + rng.normal(0, sigma, image.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def crop_resize(image: np.ndarray, keep: float = 0.85) -> np.ndarray:
    h, w = image.shape
    dh, dw = int(h * (1 - keep) / 2), int(w * (1 - keep) / 2)
    cropped = Image.fromarray(image[dh:h - dh, dw:w - dw])
    return np.array(cropped.resize((w, h), Image.BILINEAR))


def run_demo() -> None:
    message = "SYNTHID-DEMO-2026-10-09"
    seed = "daily-experiment"
    base = make_test_image()
    watermarked = {
        "lsb": embed_lsb(base, message, seed=seed),
        "dct": embed_dct(base, message, seed=seed),
    }
    transforms = {
        "clean": lambda im: im,
        "jpeg q=90": lambda im: jpeg_recompress(im, 90),
        "jpeg q=75": lambda im: jpeg_recompress(im, 75),
        "jpeg q=50": lambda im: jpeg_recompress(im, 50),
        "gaussian noise s=2": lambda im: add_noise(im, 2.0),
        "gaussian noise s=5": lambda im: add_noise(im, 5.0),
        "center crop 85%": lambda im: crop_resize(im, 0.85),
    }
    print(f"message: {message!r} | seed: {seed!r} | "
          f"threshold: {DETECT_THRESHOLD}")
    print(f"{'transform':<20} {'lsb score':>10} {'lsb hit':>8} "
          f"{'dct score':>10} {'dct hit':>8}")
    for name, fn in transforms.items():
        row = []
        for scheme in ("lsb", "dct"):
            score, hit = detect(fn(watermarked[scheme]), message,
                                 scheme=scheme, seed=seed)
            row.append((score, hit))
        (ls, lh), (ds, dh) = row
        print(f"{name:<20} {ls:>10.3f} {str(lh):>8} {ds:>10.3f} {str(dh):>8}")
    # Sanity: a wrong key must not detect.
    s, h = detect(watermarked["dct"], message, scheme="dct", seed="wrong-key")
    print(f"wrong-key check: dct score={s:.3f} detected={h} (expect ~0.5/False)")


if __name__ == "__main__":
    run_demo()
