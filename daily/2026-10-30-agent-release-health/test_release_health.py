"""Tests for release_health.py. Run: python3 -m unittest test_release_health -v"""

import unittest

from release_health import (
    MIN_SAMPLE,
    compare_releases,
    error_mix,
    group_traces,
    latency_summary,
    percentile,
    success_rate,
    two_proportion_pvalue,
)


def trace(release="v1", success=True, latency=100.0, error=None, route="chat"):
    return {"trace_id": "t", "release": release, "success": success,
            "latency_ms": latency, "error": error, "route": route}


def release(n, successes, latency=100.0):
    return [trace(success=(i < successes), latency=latency) for i in range(n)]


class TestGroupingAndRates(unittest.TestCase):
    def test_group_traces_by_release_metadata(self):
        traces = [trace("v1"), trace("v2"), trace("v1"), {"success": True}]
        grouped = group_traces(traces)
        self.assertEqual(len(grouped["v1"]), 2)
        self.assertEqual(len(grouped["v2"]), 1)

    def test_success_rate(self):
        self.assertAlmostEqual(success_rate(release(10, 7)), 0.7)
        self.assertEqual(success_rate([]), 0.0)

    def test_percentile_nearest_rank(self):
        self.assertEqual(percentile([10, 20, 30, 40], 50), 20)
        self.assertEqual(percentile([10, 20, 30, 40], 95), 40)

    def test_error_mix_shares_are_over_all_traces(self):
        traces = [trace(success=False, error="timeout")] * 10 + [trace()] * 90
        self.assertAlmostEqual(error_mix(traces)["timeout"], 0.10)


class TestStatistics(unittest.TestCase):
    def test_identical_proportions_are_not_significant(self):
        self.assertAlmostEqual(two_proportion_pvalue(90, 100, 90, 100), 1.0)

    def test_large_drop_is_significant(self):
        self.assertLess(two_proportion_pvalue(95, 100, 70, 100), 0.001)

    def test_latency_summary(self):
        summary = latency_summary([trace(latency=100.0), trace(latency=300.0)])
        self.assertEqual(summary["mean"], 200.0)
        self.assertEqual(summary["p95"], 300.0)


class TestVerdicts(unittest.TestCase):
    def test_identical_releases_are_no_regression(self):
        result = compare_releases(release(100, 95), release(100, 95))
        self.assertEqual(result["verdict"], "NO REGRESSION")

    def test_success_drop_is_a_regression(self):
        result = compare_releases(release(100, 95), release(100, 70))
        self.assertEqual(result["verdict"], "REGRESSION")
        self.assertTrue(any("success rate fell" in r for r in result["reasons"]))

    def test_latency_jump_is_a_regression(self):
        result = compare_releases(release(100, 95, latency=100.0),
                                  release(100, 95, latency=300.0))
        self.assertEqual(result["verdict"], "REGRESSION")
        self.assertTrue(any("latency rose" in r for r in result["reasons"]))

    def test_new_error_type_is_a_regression(self):
        baseline = [trace(success=(i < 95)) for i in range(100)]
        candidate = [trace(success=False, error="timeout") if i < 10
                     else trace(success=True) for i in range(100)]
        result = compare_releases(baseline, candidate)
        self.assertEqual(result["verdict"], "REGRESSION")
        self.assertTrue(any("timeout" in r for r in result["reasons"]))

    def test_small_sample_is_inconclusive(self):
        result = compare_releases(release(10, 10), release(10, 5))
        self.assertEqual(result["verdict"], "INCONCLUSIVE")

    def test_min_sample_boundary(self):
        # Exactly MIN_SAMPLE traces on each side is enough to judge.
        result = compare_releases(release(MIN_SAMPLE, MIN_SAMPLE),
                                  release(MIN_SAMPLE, MIN_SAMPLE))
        self.assertNotEqual(result["verdict"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
