"""Unit tests for the SynthID-style watermarking demo.

Run: python3 test_watermark.py
"""

import numpy as np

from watermark import (
    DETECT_THRESHOLD,
    add_noise,
    bits_to_text,
    crop_resize,
    detect,
    embed_dct,
    embed_lsb,
    extract_dct,
    extract_lsb,
    jpeg_recompress,
    make_test_image,
    text_to_bits,
)

MESSAGE = "SYNTHID-DEMO-2026-10-09"
SEED = "daily-experiment"
IMG = make_test_image()


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def test_text_bits_roundtrip():
    for text in ["hi", MESSAGE, "watermark: 123, symbols! @#%"]:
        check(f"text<->bits roundtrip {text!r}", bits_to_text(text_to_bits(text)) == text)


def test_lsb_clean_roundtrip():
    wm = embed_lsb(IMG, MESSAGE, seed=SEED)
    score, hit = detect(wm, MESSAGE, scheme="lsb", seed=SEED)
    check("lsb clean: score 1.0 and detected", score == 1.0 and hit)


def test_lsb_wrong_key_rejected():
    wm = embed_lsb(IMG, MESSAGE, seed=SEED)
    score, hit = detect(wm, MESSAGE, scheme="lsb", seed="someone-elses-key")
    check("lsb wrong key: not detected", not hit and score < 0.65)


def test_lsb_wrong_message_rejected():
    wm = embed_lsb(IMG, MESSAGE, seed=SEED)
    score, hit = detect(wm, "A TOTALLY DIFFERENT MESSAGE", scheme="lsb", seed=SEED)
    check("lsb wrong message: not detected", not hit)


def test_lsb_fragile_under_jpeg():
    # LSB watermarks die under JPEG: this is WHY transform-domain
    # watermarks exist. The demo asserts the failure, honestly.
    wm = embed_lsb(IMG, MESSAGE, seed=SEED)
    score, hit = detect(jpeg_recompress(wm, 75), MESSAGE, scheme="lsb", seed=SEED)
    check("lsb dies under jpeg q=75 (not detected)", not hit and score < DETECT_THRESHOLD)


def test_dct_clean_roundtrip():
    wm = embed_dct(IMG, MESSAGE, seed=SEED)
    score, hit = detect(wm, MESSAGE, scheme="dct", seed=SEED)
    check("dct clean: score 1.0 and detected", score == 1.0 and hit)


def test_dct_survives_jpeg():
    wm = embed_dct(IMG, MESSAGE, seed=SEED)
    for q in (90, 75):
        score, hit = detect(jpeg_recompress(wm, q), MESSAGE, scheme="dct", seed=SEED)
        check(f"dct survives jpeg q={q}", hit and score >= DETECT_THRESHOLD)


def test_dct_survives_light_noise():
    wm = embed_dct(IMG, MESSAGE, seed=SEED)
    score, hit = detect(add_noise(wm, 3.0), MESSAGE, scheme="dct", seed=SEED)
    check("dct survives gaussian noise sigma=3", hit)


def test_dct_wrong_key_rejected():
    wm = embed_dct(IMG, MESSAGE, seed=SEED)
    score, hit = detect(wm, MESSAGE, scheme="dct", seed="wrong-key")
    check("dct wrong key: not detected", not hit and score < 0.65)


def test_embed_is_deterministic():
    a = embed_dct(IMG, MESSAGE, seed=SEED)
    b = embed_dct(IMG, MESSAGE, seed=SEED)
    check("embed deterministic (same seed, same bytes)", np.array_equal(a, b))


def test_embed_does_not_mutate_input():
    before = IMG.copy()
    embed_lsb(IMG, MESSAGE, seed=SEED)
    embed_dct(IMG, MESSAGE, seed=SEED)
    check("embed leaves the input image untouched", np.array_equal(IMG, before))


def test_capacity_limits_enforced():
    too_long = "x" * 200  # 1616 bits > 1024 blocks on a 256x256 image
    try:
        embed_dct(IMG, too_long, seed=SEED)
        check("dct rejects oversize message", False)
    except ValueError:
        check("dct rejects oversize message", True)
    try:
        embed_lsb(np.zeros((8, 8), dtype=np.uint8), MESSAGE, seed=SEED)
        check("lsb rejects oversize message", False)
    except ValueError:
        check("lsb rejects oversize message", True)


def test_extract_bit_counts():
    wm = embed_lsb(IMG, MESSAGE, seed=SEED)
    n = len(text_to_bits(MESSAGE))
    check("lsb extract returns exactly n bits", len(extract_lsb(wm, n, seed=SEED)) == n)
    wmd = embed_dct(IMG, MESSAGE, seed=SEED)
    check("dct extract returns exactly n bits", len(extract_dct(wmd, n, seed=SEED)) == n)


def test_crop_is_honest_failure():
    # Heavy geometric transforms break block alignment; the demo
    # documents this as a limitation rather than hiding it.
    wm = embed_dct(IMG, MESSAGE, seed=SEED)
    score, _ = detect(crop_resize(wm, 0.85), MESSAGE, scheme="dct", seed=SEED)
    check("crop 85% degrades dct below clean 1.0 (documented limit)", score < 1.0)


if __name__ == "__main__":
    test_text_bits_roundtrip()
    test_lsb_clean_roundtrip()
    test_lsb_wrong_key_rejected()
    test_lsb_wrong_message_rejected()
    test_lsb_fragile_under_jpeg()
    test_dct_clean_roundtrip()
    test_dct_survives_jpeg()
    test_dct_survives_light_noise()
    test_dct_wrong_key_rejected()
    test_embed_is_deterministic()
    test_embed_does_not_mutate_input()
    test_capacity_limits_enforced()
    test_extract_bit_counts()
    test_crop_is_honest_failure()
    print("\nAll tests passed.")
