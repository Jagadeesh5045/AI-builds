"""Tests for the FP8 E4M3 codec, KV cache and evaluation harness."""

import math
import unittest

from fp8_kv_cache import (KVCache, cache_memory_bytes, decode_fp8_e4m3,
                          dequantize_f16, dequantize_fp8, encode_fp8_e4m3,
                          evaluate, quantize_f16, quantize_fp8)


class TestCodec(unittest.TestCase):
    def test_decode_known_values(self):
        self.assertEqual(decode_fp8_e4m3(0x00), 0.0)
        self.assertEqual(decode_fp8_e4m3(0x38), 1.0)     # exp 7, mant 0
        self.assertEqual(decode_fp8_e4m3(0xC0), -2.0)    # sign, exp 8
        self.assertEqual(decode_fp8_e4m3(0x7E), 448.0)   # max finite
        self.assertEqual(decode_fp8_e4m3(0x01), 0.001953125)  # min subnormal

    def test_decode_nan_code(self):
        self.assertTrue(math.isnan(decode_fp8_e4m3(0x7F)))
        self.assertTrue(math.isnan(decode_fp8_e4m3(0xFF)))

    def test_encode_exact_for_representable_values(self):
        for value in (0.0, 1.0, -2.0, 0.5, -0.25, 448.0, 0.001953125):
            self.assertEqual(decode_fp8_e4m3(encode_fp8_e4m3(value)), value)

    def test_encode_rounds_to_nearest(self):
        # 1.04 is closer to 1.0 (step 0.125 at that scale) than to 1.125
        self.assertEqual(decode_fp8_e4m3(encode_fp8_e4m3(1.04)), 1.0)
        self.assertEqual(decode_fp8_e4m3(encode_fp8_e4m3(1.10)), 1.125)

    def test_encode_clips_and_handles_nan(self):
        self.assertEqual(encode_fp8_e4m3(10000.0), 0x7E)
        self.assertEqual(encode_fp8_e4m3(-10000.0), 0xFE)
        self.assertEqual(encode_fp8_e4m3(math.inf), 0x7E)
        self.assertTrue(math.isnan(
            decode_fp8_e4m3(encode_fp8_e4m3(math.nan))))

    def test_decode_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            decode_fp8_e4m3(256)

    def test_quantize_roundtrip_error_bounded(self):
        values = [0.1, -0.7, 3.3, 12.0, -100.0, 0.001]
        back = dequantize_fp8(quantize_fp8(values))
        for orig, rec in zip(values, back):
            # E4M3 has 3 mantissa bits: relative error at most ~1/16
            # for normals; small absolute allowance for subnormals.
            self.assertLessEqual(abs(orig - rec),
                                 max(abs(orig) / 16.0, 0.002))

    def test_f16_roundtrip(self):
        values = [0.1, -2.5, 100.0]
        back = dequantize_f16(quantize_f16(values))
        for orig, rec in zip(values, back):
            self.assertAlmostEqual(orig, rec, places=3)


class TestKVCache(unittest.TestCase):
    def test_append_len_and_memory_ratio(self):
        dim = 32
        fp8 = KVCache(dim, "fp8")
        f16 = KVCache(dim, "f16")
        vec = [0.01 * i for i in range(dim)]
        for _ in range(10):
            fp8.append(vec, vec)
            f16.append(vec, vec)
        self.assertEqual(len(fp8), 10)
        self.assertEqual(len(f16), 10)
        self.assertEqual(f16.memory_bytes, 2 * fp8.memory_bytes)

    def test_retrieval_is_close_for_fp8_exact_for_small_f16(self):
        cache = KVCache(4, "fp8")
        cache.append([1.0, 0.5, -2.0, 0.25], [0.0, 1.0, -1.0, 2.0])
        self.assertEqual(cache.key(0), [1.0, 0.5, -2.0, 0.25])
        self.assertEqual(cache.value(0), [0.0, 1.0, -1.0, 2.0])

    def test_wrong_width_rejected(self):
        cache = KVCache(4, "fp8")
        with self.assertRaises(ValueError):
            cache.append([1.0, 2.0], [1.0, 2.0, 3.0, 4.0])

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            KVCache(4, "int4")

    def test_memory_formula_halves(self):
        args = dict(n_tokens=32768, n_layers=32, n_kv_heads=8, head_dim=128)
        self.assertEqual(cache_memory_bytes(**args, bytes_per_element=1) * 2,
                         cache_memory_bytes(**args, bytes_per_element=2))


class TestEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = evaluate(seed=7, n_tokens=128, head_dim=32,
                               n_queries=16)

    def test_fp8_uses_half_the_memory(self):
        self.assertEqual(self.results["fp8"]["memory_bytes"] * 2,
                         self.results["f16"]["memory_bytes"])

    def test_fp8_error_is_small_but_larger_than_f16(self):
        self.assertLess(self.results["fp8"]["relative_rmse"], 0.05)
        self.assertLess(self.results["f16"]["relative_rmse"],
                        self.results["fp8"]["relative_rmse"])

    def test_attention_mostly_agrees(self):
        self.assertGreaterEqual(self.results["fp8"]["top1_agreement"], 0.8)
        self.assertGreater(self.results["fp8"]["mean_cosine"], 0.99)
        self.assertLess(self.results["fp8"]["mean_kl"], 0.05)


if __name__ == "__main__":
    unittest.main()
