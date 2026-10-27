"""Tests for the LFM3 tri-hybrid mixer. Run with: python test_tri_hybrid.py"""

import unittest

import numpy as np

from tri_hybrid import (
    Attention,
    GatedDepthwiseConv,
    TriHybridStack,
    attention_flops,
    build_layer_pattern,
)


class TestTriHybrid(unittest.TestCase):
    def test_layer_pattern_counts(self):
        pattern = build_layer_pattern()
        self.assertEqual(len(pattern), 24)
        self.assertEqual(pattern.count("conv"), 13)
        self.assertEqual(pattern.count("window"), 5)
        self.assertEqual(pattern.count("global"), 6)

    def test_conv_is_causal_and_local(self):
        conv = GatedDepthwiseConv(d_model=4, kernel=7,
                                  rng=np.random.default_rng(0))
        x = np.zeros((16, 4))
        x[3] = [1.0, 0.0, 0.0, 0.0]
        out = conv.forward(x)
        affected = [t for t in range(16)
                    if np.linalg.norm(out[t]) > 1e-9]
        self.assertEqual(affected, [3, 4, 5, 6, 7, 8, 9])

    def test_conv_zero_input_gives_zero_output(self):
        conv = GatedDepthwiseConv(d_model=4, kernel=7,
                                  rng=np.random.default_rng(1))
        out = conv.forward(np.zeros((8, 4)))
        np.testing.assert_array_equal(out, np.zeros((8, 4)))

    def test_attention_is_causal(self):
        attn = Attention(4, mode="global", rng=np.random.default_rng(2))
        x = np.random.default_rng(3).normal(size=(10, 4))
        weights = attn.weights(x)
        # No query attends to a future key.
        self.assertTrue(np.all(weights[np.triu_indices(10, k=1)] == 0.0))
        # Every query's weights sum to one.
        np.testing.assert_allclose(weights.sum(axis=1),
                                   np.ones(10), atol=1e-9)

    def test_window_attention_respects_window(self):
        attn = Attention(4, mode="window", window=4,
                         rng=np.random.default_rng(4))
        x = np.random.default_rng(5).normal(size=(12, 4))
        weights = attn.weights(x)
        for t in range(12):
            for j in range(12):
                if j < t - 3 or j > t:
                    self.assertEqual(weights[t, j], 0.0)

    def test_stack_shape_and_finiteness(self):
        stack = TriHybridStack(d_model=16, seed=0)
        x = np.random.default_rng(6).normal(size=(32, 16))
        out = stack.forward(x)
        self.assertEqual(out.shape, (32, 16))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_stack_is_causal(self):
        stack = TriHybridStack(d_model=16, seed=0)
        x = np.random.default_rng(7).normal(size=(32, 16))
        x2 = x.copy()
        x2[-1] += 10.0
        out, out2 = stack.forward(x), stack.forward(x2)
        np.testing.assert_allclose(out[:-1], out2[:-1], atol=1e-9)
        self.assertGreater(np.max(np.abs(out[-1] - out2[-1])), 0.0)

    def test_hybrid_is_cheaper_than_full_attention(self):
        full, hybrid, ratio = attention_flops(262144, d_model=128)
        self.assertLess(hybrid, full)
        # Only 6 of 24 layers are quadratic, so the share tends to 0.25.
        self.assertGreater(ratio, 0.20)
        self.assertLess(ratio, 0.30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
