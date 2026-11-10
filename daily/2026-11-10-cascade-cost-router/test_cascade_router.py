"""Tests for the cascade cost router. Pure stdlib unittest."""

import unittest

from cascade_router import (
    CHEAP, MID, FRONTIER, CASES, CascadeRouter,
    cheap_predict, mid_predict, frontier_predict,
    estimate_input_tokens, evaluate_cascade, evaluate_single_tier,
)


class TestTierPricing(unittest.TestCase):
    def test_reported_price_ordering(self):
        # Cheap tier is 20x cheaper than mid on input, 80x vs frontier.
        self.assertEqual(MID.input_per_million / CHEAP.input_per_million, 20.0)
        self.assertEqual(FRONTIER.input_per_million / CHEAP.input_per_million, 80.0)

    def test_cost_scales_linearly(self):
        self.assertAlmostEqual(CHEAP.cost(1_000_000, 0), 0.10)
        self.assertAlmostEqual(FRONTIER.cost(0, 1_000_000), 40.00)

    def test_token_estimate_grows_with_text(self):
        self.assertGreater(estimate_input_tokens("one two three four"),
                           estimate_input_tokens("one"))


class TestTierAccuracyByBand(unittest.TestCase):
    def _band_accuracy(self, predict, band):
        cases = [c for c in CASES if c.band == band]
        return sum(1 for c in cases if predict(c.text).label == c.gold), len(cases)

    def test_cheap_gets_all_clear_cases(self):
        ok, total = self._band_accuracy(cheap_predict, "clear")
        self.assertEqual(ok, total)

    def test_cheap_struggles_off_the_clear_band(self):
        ok, total = self._band_accuracy(cheap_predict, "paraphrase")
        self.assertLess(ok, total)
        ok, total = self._band_accuracy(cheap_predict, "hard")
        self.assertLess(ok, total)

    def test_mid_recovers_all_paraphrases(self):
        ok, total = self._band_accuracy(mid_predict, "paraphrase")
        self.assertEqual(ok, total)

    def test_frontier_gets_everything(self):
        for band in ("clear", "paraphrase", "hard"):
            ok, total = self._band_accuracy(frontier_predict, band)
            self.assertEqual(ok, total, band)

    def test_negation_only_handled_at_frontier(self):
        # Probe where negation alone flips the answer: both churn words
        # are negated, so the real request is the bug report. The cheap
        # tier has no negation rule and misreads it as churn_risk.
        text = "do not cancel and do not leave us, just fix the error"
        self.assertEqual(cheap_predict(text).label, "churn_risk")  # misread
        self.assertEqual(frontier_predict(text).label, "bug_report")


class TestCascade(unittest.TestCase):
    def setUp(self):
        self.router = CascadeRouter()

    def test_clear_case_stays_cheap(self):
        result = self.router.route("refund the payment on my last invoice")
        self.assertEqual(result.tier_used, "cheap")
        self.assertEqual(result.escalations, 0)
        self.assertEqual(result.label, "billing")

    def test_paraphrase_escalates_to_mid(self):
        result = self.router.route("extract the numbers into a spreadsheet dump")
        self.assertEqual(result.tier_used, "mid")
        self.assertEqual(result.label, "data_export")

    def test_hard_case_reaches_frontier(self):
        result = self.router.route(
            "intermittent exception in the stack, a regression I can reproduce")
        self.assertEqual(result.tier_used, "frontier")
        self.assertEqual(result.label, "bug_report")

    def test_escalated_cost_includes_every_tier_visited(self):
        result = self.router.route(
            "intermittent exception in the stack, a regression I can reproduce")
        self.assertEqual(len(result.confidence_path), 3)
        single = FRONTIER.cost(
            estimate_input_tokens(
                "intermittent exception in the stack, a regression I can reproduce"),
            12)
        # Cascade pays cheap + mid + frontier, so it costs MORE than a
        # direct frontier call for this one case. The saving is across
        # the workload, never per escalated case.
        self.assertGreater(result.total_cost, single)

    def test_cascade_matches_frontier_accuracy_for_less(self):
        cascade, counts = evaluate_cascade(self.router)
        frontier = evaluate_single_tier("frontier only", frontier_predict, FRONTIER)
        self.assertEqual(cascade.accuracy, frontier.accuracy)
        self.assertLess(cascade.total_cost, frontier.total_cost)
        self.assertEqual(sum(counts.values()), len(CASES))
        self.assertGreater(counts["cheap"], 0)

    def test_threshold_extremes(self):
        # Threshold 0: nothing ever escalates, cascade == cheap accuracy.
        report, _ = evaluate_cascade(CascadeRouter(threshold=0.0))
        cheap = evaluate_single_tier("cheap only", cheap_predict, CHEAP)
        self.assertEqual(report.accuracy, cheap.accuracy)
        # Very high threshold: everything reaches the frontier tier.
        _, counts = evaluate_cascade(CascadeRouter(threshold=0.99))
        self.assertEqual(counts["frontier"], len(CASES))


if __name__ == "__main__":
    unittest.main()
