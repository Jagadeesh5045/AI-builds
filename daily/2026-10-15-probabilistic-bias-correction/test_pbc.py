#!/usr/bin/env python3
"""Tests for pbc.py. All deterministic (fixed seeds); stdlib unittest only."""

import math
import random
import unittest

from pbc import (
    BiasedModel,
    PBCorrector,
    Phi,
    World,
    brier_score,
    crps_cdf,
    crps_gaussian_closed,
    gaussian_cdf,
    gaussian_exceedance,
    phi,
    pit_chi_square,
    run_demo,
)


class TestGaussianHelpers(unittest.TestCase):
    def test_phi_known_value(self):
        self.assertAlmostEqual(phi(0.0), 1.0 / math.sqrt(2.0 * math.pi))

    def test_Phi_known_values(self):
        self.assertAlmostEqual(Phi(0.0), 0.5)
        self.assertAlmostEqual(Phi(2.0), 0.9772498680518208, places=9)
        self.assertAlmostEqual(Phi(-2.0), 1.0 - 0.9772498680518208, places=9)

    def test_gaussian_cdf_matches_Phi(self):
        self.assertAlmostEqual(gaussian_cdf(1.5, 1.5, 2.0), 0.5)
        self.assertAlmostEqual(gaussian_cdf(5.5, 1.5, 2.0), Phi(2.0))

    def test_gaussian_cdf_degenerate_sigma(self):
        self.assertEqual(gaussian_cdf(3.0, 3.0, 0.0), 1.0)
        self.assertEqual(gaussian_cdf(2.9, 3.0, 0.0), 0.0)

    def test_exceedance_complements_cdf(self):
        self.assertAlmostEqual(
            gaussian_exceedance(5.0, 1.0, 2.0) + gaussian_cdf(5.0, 1.0, 2.0), 1.0)


class TestCRPS(unittest.TestCase):
    def test_closed_form_matches_numerical(self):
        rng = random.Random(123)
        for _ in range(10):
            mu = rng.uniform(-10, 10)
            sigma = rng.uniform(0.5, 5.0)
            y = rng.uniform(-15, 15)
            closed = crps_gaussian_closed(mu, sigma, y)
            numeric = crps_cdf(lambda x: gaussian_cdf(x, mu, sigma),
                               y, mu - 10 * sigma, mu + 10 * sigma, n=4001)
            self.assertAlmostEqual(closed, numeric, places=3)

    def test_crps_degenerate_is_absolute_error(self):
        self.assertAlmostEqual(crps_gaussian_closed(2.0, 0.0, 5.0), 3.0)

    def test_crps_nonnegative(self):
        rng = random.Random(9)
        for _ in range(20):
            mu, sigma = rng.uniform(-5, 5), rng.uniform(0.2, 4.0)
            y = rng.uniform(-8, 8)
            self.assertGreaterEqual(crps_gaussian_closed(mu, sigma, y), 0.0)

    def test_crps_perfect_forecast_is_zero(self):
        self.assertAlmostEqual(crps_gaussian_closed(4.0, 1e-9, 4.0), 0.0, places=6)

    def test_crps_prefers_sharper_calibrated(self):
        # Same mean, narrower spread around the truth -> lower CRPS.
        y = 1.0
        self.assertLess(crps_gaussian_closed(1.0, 0.5, y),
                        crps_gaussian_closed(1.0, 3.0, y))


class TestPBCorrector(unittest.TestCase):
    def _archive(self, n=600, seed=21, biased=True):
        rng = random.Random(seed)
        archive = []
        model = BiasedModel() if biased else None
        for t in range(n):
            seasonal = 8.0 * math.sin(2.0 * math.pi * t / 365.0)
            y = seasonal + rng.gauss(0.0, 2.6)
            if biased:
                mu, sigma = BiasedModel().forecast(t)
            else:
                mu, sigma = seasonal, 2.6  # well-calibrated forecaster
            archive.append((mu, sigma, y))
        return archive

    def test_fit_rejects_empty_archive(self):
        with self.assertRaises(ValueError):
            PBCorrector().fit([])

    def test_pit_cdf_bounds(self):
        c = PBCorrector().fit(self._archive())
        self.assertEqual(c.pit_cdf(-1.0), 0.0)
        self.assertEqual(c.pit_cdf(2.0), 1.0)
        self.assertEqual(c.pit_cdf(0.0), 0.0)
        self.assertEqual(c.pit_cdf(1.0), 1.0)

    def test_pit_cdf_monotone(self):
        c = PBCorrector().fit(self._archive())
        grid = [i / 100 for i in range(101)]
        vals = [c.pit_cdf(z) for z in grid]
        for a, b in zip(vals, vals[1:]):
            self.assertLessEqual(a, b + 1e-12)

    def test_corrected_cdf_is_valid_cdf(self):
        c = PBCorrector().fit(self._archive())
        f = c.corrected_cdf(2.0, 1.5)
        self.assertAlmostEqual(f(-1e6), 0.0, places=6)
        self.assertAlmostEqual(f(1e6), 1.0, places=6)
        xs = [-10 + i * 0.5 for i in range(81)]
        vals = [f(x) for x in xs]
        for a, b in zip(vals, vals[1:]):
            self.assertLessEqual(a, b + 1e-12)
        for v in vals:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_corrector_sees_only_archive(self):
        archive = self._archive(n=500)
        c = PBCorrector().fit(archive)
        self.assertEqual(c.n_fit, len(archive))  # no holdout leakage by design

    def test_correction_improves_crps_on_holdout(self):
        archive = self._archive(n=800, seed=21)
        holdout = self._archive(n=300, seed=999)
        c = PBCorrector().fit(archive)
        raw, corr = 0.0, 0.0
        for mu, sigma, y in holdout:
            raw += crps_gaussian_closed(mu, sigma, y)
            f = c.corrected_cdf(mu, sigma)
            corr += crps_cdf(f, y, mu - 8 * sigma, mu + 8 * sigma)
        self.assertLess(corr, raw)

    def test_pit_uniformity_improves(self):
        archive = self._archive(n=800, seed=21)
        holdout = self._archive(n=400, seed=555)
        c = PBCorrector().fit(archive)
        pit_raw = [gaussian_cdf(y, mu, sigma) for mu, sigma, y in holdout]
        pit_corr = [c.corrected_cdf(mu, sigma)(y) for mu, sigma, y in holdout]
        self.assertLess(pit_chi_square(pit_corr), pit_chi_square(pit_raw))


class TestDemo(unittest.TestCase):
    def test_demo_reproducible(self):
        r1 = run_demo(seed=7)
        r2 = run_demo(seed=7)
        self.assertEqual(r1, r2)

    def test_demo_pbc_beats_raw_and_climatology(self):
        r = run_demo(seed=7)
        self.assertLess(r["crps_corrected"], r["crps_raw"])
        self.assertLess(r["crps_corrected"], r["crps_climatology"])
        self.assertLess(r["brier_corrected"], r["brier_raw"])
        self.assertLess(r["pit_chi2_corrected"], r["pit_chi2_raw"])

    def test_brier_score_sane(self):
        self.assertAlmostEqual(brier_score([(0.0, 0), (1.0, 1)]), 0.0)
        self.assertAlmostEqual(brier_score([(1.0, 0), (0.0, 1)]), 1.0)

    def test_pit_chi_square_uniform_reference(self):
        rng = random.Random(4)
        pits = [rng.random() for _ in range(1000)]  # truly uniform
        self.assertLess(pit_chi_square(pits), 30.0)  # df=9, 99.9% crit ~27.9


if __name__ == "__main__":
    unittest.main()
