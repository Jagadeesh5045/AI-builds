"""Tests for wafer_serving.py. Run: python3 -m unittest test_wafer_serving -v"""

import math
import unittest

from wafer_serving import (
    hop_latency_us,
    joules_per_token,
    kv_capacity_tokens,
    layer_bytes_gb,
    pipeline_time,
    plan_wafers,
    quantize_error,
    random_matrix,
    streams_needed,
    weight_bytes_gb,
)


class TestWeightBudget(unittest.TestCase):
    def test_total_payload(self):
        # 3T params at 4 bits = 1.5e12 bytes = 1500 GB.
        self.assertAlmostEqual(weight_bytes_gb(3e12, 4), 1500.0)

    def test_layer_size_matches_post(self):
        # Cerebras: "about 16 GB" per layer over 93 layers.
        self.assertAlmostEqual(layer_bytes_gb(3e12, 4, 93), 16.13, places=2)

    def test_16bit_payload_four_times_4bit(self):
        self.assertAlmostEqual(weight_bytes_gb(3e12, 16),
                               4 * weight_bytes_gb(3e12, 4))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            weight_bytes_gb(0, 4)
        with self.assertRaises(ValueError):
            weight_bytes_gb(3e12, 0)


class TestWaferPlan(unittest.TestCase):
    def setUp(self):
        self.plan = plan_wafers(69, 24, wafer_gb=44.0,
                                layer_gb=layer_bytes_gb(3e12, 4, 93))

    def test_total_is_59(self):
        self.assertEqual(self.plan["total_wafers"], 59)

    def test_linear_packing(self):
        # 69 linear layers, 2 per wafer -> 35 wafers, last one holding 1.
        self.assertEqual(self.plan["linear_wafers"], 35)
        self.assertEqual(self.plan["last_linear_wafer_layers"], 1)

    def test_dense_packing(self):
        self.assertEqual(self.plan["dense_wafers"], 24)

    def test_dense_spare_room(self):
        # 44 - 16.13 = ~27.87 GB left on each dense wafer for KV cache.
        self.assertAlmostEqual(self.plan["spare_gb_per_dense_wafer"], 27.87,
                               places=2)

    def test_oversize_layer_rejected(self):
        with self.assertRaises(ValueError):
            plan_wafers(69, 24, wafer_gb=44.0, layer_gb=50.0)


class TestPipeline(unittest.TestCase):
    def test_more_microbatches_less_bubble(self):
        small = pipeline_time(59, 64)
        large = pipeline_time(59, 256)
        self.assertLess(large["bubble_fraction"], small["bubble_fraction"])
        self.assertGreater(large["efficiency"], small["efficiency"])

    def test_efficiency_formula(self):
        pipe = pipeline_time(59, 64)
        self.assertAlmostEqual(pipe["efficiency"], 64 / (64 + 58))
        self.assertAlmostEqual(pipe["efficiency"] + pipe["bubble_fraction"],
                               1.0)

    def test_hop_latency(self):
        # 58 gaps x 2 us = 116 us across the whole 59-wafer pipeline.
        self.assertEqual(hop_latency_us(59), 116.0)


class TestQuantisation(unittest.TestCase):
    def test_error_falls_with_bits(self):
        matrix = random_matrix(64, 64, seed=42)
        e4 = quantize_error(matrix, 4)[0]
        e8 = quantize_error(matrix, 8)[0]
        e16 = quantize_error(matrix, 16)[0]
        self.assertGreater(e4, e8)
        self.assertGreater(e8, e16)
        self.assertLess(e4, 0.2)  # 4-bit on Gaussian weights: coarse, not fatal

    def test_zero_matrix_is_exact(self):
        err, _ = quantize_error([[0.0, 0.0]], 4)
        self.assertEqual(err, 0.0)


class TestEnergy(unittest.TestCase):
    def test_joules_per_token(self):
        self.assertAlmostEqual(joules_per_token(1.18e6, 1.18e6 / 4.1), 4.1)

    def test_stream_reconciliation(self):
        # ~288k aggregate tok/s / 1,500 tok/s per stream ~= 192 streams.
        streams = streams_needed(1.18e6, 4.1, 1500)
        self.assertAlmostEqual(streams, 191.9, places=0)
        self.assertGreater(streams_needed(1.18e6, 4.1, 2000), 140)

    def test_kv_capacity_scales_with_assumption(self):
        # Illustrative only: 8-bit KV, 16 KB per token per dense layer.
        cap = kv_capacity_tokens(24, 27.87, 16 * 1024)
        self.assertGreater(cap, 0)
        self.assertEqual(kv_capacity_tokens(24, 27.87, 32 * 1024),
                         cap // 2 if cap % 2 == 0 else cap // 2 + 1 // 2)


if __name__ == "__main__":
    unittest.main()
