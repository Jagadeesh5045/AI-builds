"""Tests for gguf_quant: exact memory maths, code ranges, error ordering."""
import unittest

import numpy as np

from gguf_quant import (MODEL_PARAMS, dequantize, effective_bits_per_weight,
                        matvec_metrics, memory_table, quantize,
                        roundtrip_metrics, storage_bytes)


class TestMemoryMaths(unittest.TestCase):
    def test_fp16_baseline_exact(self):
        # 35B params at 2 bytes each is exactly 70 GB (decimal).
        self.assertEqual(storage_bytes(MODEL_PARAMS, None), 70_000_000_000)

    def test_q8_exact(self):
        # codes 35 GB + ceil(35e9/32) blocks * 2 scale bytes.
        self.assertEqual(storage_bytes(MODEL_PARAMS, 8), 37_187_500_000)

    def test_q4_exact(self):
        # packed codes 17.5 GB + the same scale overhead as Q8.
        self.assertEqual(storage_bytes(MODEL_PARAMS, 4), 19_687_500_000)

    def test_effective_bits_per_weight(self):
        self.assertAlmostEqual(effective_bits_per_weight(4), 4.5)
        self.assertAlmostEqual(effective_bits_per_weight(8), 8.5)

    def test_only_q4_and_fp16_verdicts_for_32gb(self):
        table = {r["format"]: r for r in memory_table()}
        self.assertFalse(table["FP16"]["fits_32gb_ram"])
        self.assertFalse(table["Q8_0-style"]["fits_32gb_ram"])
        self.assertTrue(table["Q4_0-style"]["fits_32gb_ram"])

    def test_q4_more_than_3x_smaller_than_fp16(self):
        ratio = storage_bytes(MODEL_PARAMS, None) / storage_bytes(MODEL_PARAMS, 4)
        self.assertGreater(ratio, 3.5)


class TestQuantizer(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = rng.standard_normal((64, 48)) / 8.0

    def test_roundtrip_shape_preserved(self):
        self.assertEqual(dequantize(quantize(self.w, 4)).shape, self.w.shape)

    def test_codes_within_signed_range(self):
        q4 = quantize(self.w, 4)
        self.assertGreaterEqual(int(q4.codes.min()), -8)
        self.assertLessEqual(int(q4.codes.max()), 7)
        q8 = quantize(self.w, 8)
        self.assertGreaterEqual(int(q8.codes.min()), -128)
        self.assertLessEqual(int(q8.codes.max()), 127)

    def test_zero_block_dequantises_to_zero(self):
        z = np.zeros((4, 32))
        np.testing.assert_array_equal(dequantize(quantize(z, 4)), z)

    def test_largest_weight_in_block_is_near_exact(self):
        # The max-magnitude weight defines the scale, so it maps to
        # +/-qmax and comes back (near) exactly.
        block = np.zeros((1, 32)); block[0, 5] = 3.25
        back = dequantize(quantize(block, 4))
        self.assertAlmostEqual(float(back[0, 5]), 3.25, places=9)

    def test_unsupported_bits_rejected(self):
        with self.assertRaises(ValueError):
            quantize(self.w, 3)


class TestErrorBehaviour(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = rng.standard_normal((256, 256)) / 16.0

    def test_q8_error_below_q4_error(self):
        m8 = roundtrip_metrics(self.w, 8)
        m4 = roundtrip_metrics(self.w, 4)
        self.assertLess(m8["mse"], m4["mse"])
        self.assertLess(m8["relative_l2_error"], m4["relative_l2_error"])

    def test_matvec_outputs_stay_close(self):
        by_bits = {m["bits"]: m for m in matvec_metrics()}
        self.assertGreater(by_bits[8]["output_cosine"], 0.999)
        self.assertGreater(by_bits[4]["output_cosine"], 0.98)
        self.assertLess(by_bits[4]["output_rel_error"], 0.15)


if __name__ == "__main__":
    unittest.main()
