"""Tests for router.py. Stdlib unittest only (no pytest on this VM)."""

import unittest

from router import (
    STANDARD, HIGHSPEED, Task, ModelOutput, route, cost, latency_seconds,
    check_output, generation_cost_factor, compare_k3_k31,
)


class TestPricing(unittest.TestCase):
    def test_standard_cost(self):
        # 10k input, 5k output: 0.01 * $2 + 0.005 * $10 = $0.07
        self.assertAlmostEqual(cost(STANDARD, 10_000, 5_000), 0.07)

    def test_highspeed_cost_is_double_on_uncached(self):
        self.assertAlmostEqual(cost(HIGHSPEED, 10_000, 5_000), 0.14)

    def test_cached_input_costs_the_same_in_both_tiers(self):
        # Fully cached 10k input + 5k output: $0.002 cache + output price.
        self.assertAlmostEqual(cost(STANDARD, 10_000, 5_000, 10_000), 0.052)
        self.assertAlmostEqual(cost(HIGHSPEED, 10_000, 5_000, 10_000), 0.102)

    def test_invalid_token_counts_rejected(self):
        with self.assertRaises(ValueError):
            cost(STANDARD, 10_000, 5_000, 20_000)  # cached > input
        with self.assertRaises(ValueError):
            cost(STANDARD, -1, 5_000)


class TestLatency(unittest.TestCase):
    def test_highspeed_is_faster(self):
        self.assertLess(latency_seconds(HIGHSPEED, 10_000, 1_000),
                        latency_seconds(STANDARD, 10_000, 1_000))

    def test_highspeed_latency_is_standard_divided_by_speed(self):
        self.assertAlmostEqual(
            latency_seconds(HIGHSPEED, 0, 10_000),
            latency_seconds(STANDARD, 0, 10_000) / 2.5)


class TestRouting(unittest.TestCase):
    def test_interactive_goes_highspeed(self):
        task = Task("coding", 10_000, 3_000, urgency=0.9)
        self.assertEqual(route(task).variant, HIGHSPEED)

    def test_batch_goes_standard(self):
        task = Task("coding", 10_000, 3_000, urgency=0.0)
        self.assertEqual(route(task).variant, STANDARD)

    def test_tight_deadline_goes_highspeed(self):
        # Standard: 10k out -> 6.0s; Highspeed: 2.4s; deadline 3s.
        task = Task("agent", 0, 10_000, urgency=0.5, deadline_seconds=3)
        self.assertEqual(route(task).variant, HIGHSPEED)

    def test_cache_heavy_moderate_urgency_goes_highspeed(self):
        task = Task("cad", 100_000, 5_000, 80_000, urgency=0.4)
        self.assertEqual(route(task).variant, HIGHSPEED)

    def test_cache_heavy_low_urgency_stays_standard(self):
        task = Task("cad", 100_000, 5_000, 80_000, urgency=0.1)
        self.assertEqual(route(task).variant, STANDARD)


class TestGate(unittest.TestCase):
    def test_good_output_passes(self):
        out = ModelOutput("coding", "def f(): return 1",
                          {"language": "python", "code": "def f(): return 1"})
        self.assertTrue(check_output(out).passed)

    def test_missing_field_fails(self):
        out = ModelOutput("finance", "summary", {"ticker": "ACME"})
        result = check_output(out)
        self.assertFalse(result.passed)
        self.assertIn("missing required field: total", result.reasons)

    def test_placeholder_fails(self):
        out = ModelOutput("cad", "TODO later",
                          {"units": "mm", "dimensions": "10x10"})
        self.assertFalse(check_output(out).passed)

    def test_numeric_mismatch_fails_but_tolerance_passes(self):
        bad = ModelOutput("finance", "total 100",
                          {"ticker": "ACME", "total": "100"},
                          numeric_checks=[("revenue", 100.0, 118.0, 0.01)])
        self.assertFalse(check_output(bad).passed)
        good = ModelOutput("finance", "total 100",
                           {"ticker": "ACME", "total": "100"},
                           numeric_checks=[("revenue", 100.0, 100.005, 0.01)])
        self.assertTrue(check_output(good).passed)


class TestPriceVsTaskCost(unittest.TestCase):
    def test_generation_cost_factor(self):
        # (10/15) * 1.14 = 0.76, not the headline 0.67.
        self.assertAlmostEqual(generation_cost_factor(15.0, 10.0, 0.14), 0.76, places=2)

    def test_k31_still_cheaper_on_output_heavy_task(self):
        cmp_ = compare_k3_k31(2_000, 10_000)
        self.assertEqual(cmp_["k31_output_tokens"], 11_400)
        self.assertAlmostEqual(cmp_["k3_cost"], 0.156)
        self.assertAlmostEqual(cmp_["k31_cost"], 0.118)
        self.assertLess(cmp_["k31_cost"], cmp_["k3_cost"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
