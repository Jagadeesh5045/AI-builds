import numpy as np

from ternary_weights import (
    dequantize, pack_codes, size_report, sparsity, ternary_matvec,
    ternary_quantize, unpack_codes,
)


def test_codes_are_ternary():
    rng = np.random.default_rng(0)
    codes, scales = ternary_quantize(rng.normal(size=(8, 32)))
    assert set(np.unique(codes)).issubset({-1, 0, 1})
    assert np.all(scales > 0)


def test_threshold_zeroes_small_weights():
    w = np.array([[0.01, -0.02, 5.0, -5.0]])
    codes, scale = ternary_quantize(w)
    assert codes[0, 0] == 0 and codes[0, 1] == 0
    assert codes[0, 2] == 1 and codes[0, 3] == -1
    assert abs(scale - 5.0) < 1e-9


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(1)
    codes, _ = ternary_quantize(rng.normal(size=(7, 23)))  # 161 codes, needs padding
    back = unpack_codes(pack_codes(codes), codes.size)
    assert np.array_equal(back, codes.ravel())


def test_packing_density():
    packed = pack_codes(np.ones(100, dtype=np.int8))
    assert len(packed) == 20  # 5 codes per byte


def test_matvec_matches_dequantized():
    rng = np.random.default_rng(2)
    w = rng.normal(size=(16, 40))
    codes, scales = ternary_quantize(w)
    x = rng.normal(size=40)
    got = ternary_matvec(codes, scales, x)
    want = dequantize(codes, scales) @ x
    assert np.allclose(got, want, atol=1e-9)


def test_size_report_27b():
    rep = size_report(27_000_000_000)
    assert rep["fp16_bytes"] == 54_000_000_000
    assert rep["packed_bytes"] == 5_400_000_000
    assert abs(rep["packed_ratio_vs_fp16"] - 10.0) < 1e-9
    assert abs(rep["bits_per_weight_theory"] - 1.585) < 0.001


def test_sparsity_and_error_are_sane():
    rng = np.random.default_rng(3)
    w = rng.normal(size=(32, 64))
    codes, scales = ternary_quantize(w)
    assert 0.2 < sparsity(codes) < 0.8  # the 0.7*mean|w| rule zeroes a real share
    rel = np.linalg.norm(w - dequantize(codes, scales)) / np.linalg.norm(w)
    assert rel < 0.6  # lossy, but far from destroyed


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
