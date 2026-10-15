#!/usr/bin/env python3
"""Probabilistic Bias Correction (PBC) for forecasts - a toy demo.

Trending hook (14-15 Oct 2026): a Microsoft Research-led team (Hannah Guan,
Soukayna Mouatadid, Judah Cohen of MIT, Lester Mackey of MSR, with ECMWF and
other collaborators) published a Nature paper introducing *probabilistic bias
correction*: an ML framework that learns to correct a forecast model's
*historical probabilistic forecasts*, operating directly on probability
distributions rather than raw observations. Reported results: lifted accuracy
on 98% of precipitation targets and 92% of temperature targets for ECMWF's top
physics-based system, doubled the skill of ECMWF's own AI forecaster, and -
for the first time - beat climatological averages on the 2-6 week subseasonal
horizon. The authors released the code open source and note it runs on ordinary
CPUs (per MIT News / Unite.AI coverage, 14-15 Oct 2026).

This file is an INDEPENDENT, stdlib-only re-implementation of the *idea* on
synthetic data. It is not the authors' code, and the numbers below come from a
toy world, not the paper.

The core idea, in one line
--------------------------
Given a model's past forecast CDFs F_1..F_n and the outcomes y_1..y_n, the
probability-integral-transform (PIT) values z_i = F_i(y_i) should look uniform
if the model is probabilistically calibrated. Systematic deviation of the PIT
distribution from uniform *is* the model's distributional bias. PBC learns the
empirical CDF G of past PIT values and corrects each new forecast as

    F_corrected(x) = G(F_raw(x))

a correction applied directly to the probability distribution - the paper's
headline move.

Toy world
---------
Daily temperature anomaly: y_t = seasonal(t) + AR(1) noise.
A biased "dynamical" model issues Gaussian forecasts N(mu, sigma) with three
systemic flaws that mirror real NWP biases:
  1. conditional mean bias (damped extremes + warm offset + slow drift),
  2. overconfidence (spread too narrow),
  3. the bias drifts slowly over time.
PBC learns G from a 3-year archive, then we score raw vs corrected vs
climatology on a 1-year holdout using CRPS, Brier score and PIT uniformity.

Run:
    python3 pbc.py            # prints the demo scoreboard
    python3 -m unittest test_pbc -v   # runs the test suite
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


# ---------------------------------------------------------------------------
# Gaussian helpers (stdlib only)
# ---------------------------------------------------------------------------

def phi(z: float) -> float:
    """Standard normal pdf."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def Phi(z: float) -> float:
    """Standard normal cdf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def gaussian_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of N(mu, sigma) at x (degenerate point mass when sigma <= 0)."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return Phi((x - mu) / sigma)


def gaussian_exceedance(threshold: float, mu: float, sigma: float) -> float:
    """P(Y > threshold) under N(mu, sigma)."""
    return 1.0 - gaussian_cdf(threshold, mu, sigma)


def crps_gaussian_closed(mu: float, sigma: float, y: float) -> float:
    """Closed-form CRPS of N(mu, sigma) against outcome y.

    Gneiting & Raftery (2007): with z = (y - mu) / sigma,
    CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)).
    """
    if sigma <= 0:
        return abs(y - mu)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * Phi(z) - 1.0) + 2.0 * phi(z) - INV_SQRT_PI)


def _trapz(f, lo: float, hi: float, n: int) -> float:
    """Trapezoid rule on a smooth integrand."""
    if n < 2 or hi <= lo:
        return 0.0
    h = (hi - lo) / (n - 1)
    total = 0.5 * f(lo) + 0.5 * f(hi)
    for i in range(1, n - 1):
        total += f(lo + i * h)
    return total * h


def crps_cdf(cdf, y: float, lo: float, hi: float, n: int = 1001) -> float:
    """CRPS by numerical integration: integral of (F(x) - 1{x >= y})^2 dx.

    The integrand jumps at x = y, so the integral is split there and each
    smooth piece is integrated separately (otherwise the trapezoid rule is
    only first-order accurate).
    """
    if n < 3:
        raise ValueError("n must be >= 3")
    # Widen the window so it always covers the outcome y with margin: the
    # CRPS tails only vanish once x is far beyond both the forecast mass
    # and y itself.
    span = hi - lo
    lo = min(lo, y - span)
    hi = max(hi, y + span)
    if y <= lo:  # whole range is x >= y: indicator is 1
        return _trapz(lambda x: (cdf(x) - 1.0) ** 2, lo, hi, n)
    if y >= hi:  # whole range is x < y: indicator is 0
        return _trapz(lambda x: cdf(x) ** 2, lo, hi, n)
    n_left = max(2, n // 2)
    n_right = max(2, n - n_left)
    left = _trapz(lambda x: cdf(x) ** 2, lo, y, n_left)
    right = _trapz(lambda x: (cdf(x) - 1.0) ** 2, y, hi, n_right)
    return left + right


# ---------------------------------------------------------------------------
# Toy world: truth + biased "dynamical" forecaster
# ---------------------------------------------------------------------------

@dataclass
class World:
    n_days: int
    seed: int
    truth: list

    @classmethod
    def generate(cls, n_days: int = 1460, seed: int = 7) -> "World":
        rng = random.Random(seed)
        truth: list[float] = []
        noise = 0.0
        for t in range(n_days):
            seasonal = 8.0 * math.sin(2.0 * math.pi * t / 365.0)
            noise = 0.7 * noise + rng.gauss(0.0, 1.8)
            truth.append(seasonal + noise)
        return cls(n_days=n_days, seed=seed, truth=truth)


@dataclass
class BiasedModel:
    """Toy 'dynamical' forecaster with systemic distributional biases.

    The model sees the seasonal cycle but damps its amplitude, adds a warm
    offset that drifts slowly upward, and is overconfident about its spread -
    the same *kinds* of systemic bias the PBC paper targets in real NWP output.
    """

    warm_bias: float = 1.2
    damping: float = 0.75
    overconfidence: float = 0.6  # spread multiplier; <1 means overconfident
    drift_per_day: float = 0.0008
    true_cond_sd: float = 2.6

    def forecast(self, t: int) -> tuple[float, float]:
        seasonal = 8.0 * math.sin(2.0 * math.pi * t / 365.0)
        mu = self.damping * seasonal + self.warm_bias + self.drift_per_day * t
        sigma = self.overconfidence * self.true_cond_sd
        return mu, sigma


# ---------------------------------------------------------------------------
# Probabilistic Bias Correction
# ---------------------------------------------------------------------------

class PBCorrector:
    """Learns G, the empirical CDF of historical PIT values, then remaps.

    Fit on an archive of (forecast_mu, forecast_sigma, outcome) triples from
    the *past only*. Corrects a new forecast distribution via
    F_corrected(x) = G(F_raw(x)).
    """

    def __init__(self) -> None:
        self._knot_x: list[float] = []  # PIT order statistics, anchored
        self._knot_y: list[float] = []  # plotting positions, anchored
        self.n_fit: int = 0

    def fit(self, archive: list[tuple[float, float, float]]) -> "PBCorrector":
        if not archive:
            raise ValueError("archive must not be empty")
        pits = [gaussian_cdf(y, mu, sigma) for mu, sigma, y in archive]
        pits.sort()
        n = len(pits)
        # Weibull plotting positions P(Z <= xs[k]) = (k+1)/(n+1), anchored at
        # (0, 0) and (1, 1): continuous, monotone, and never exactly 0/1 inside.
        self._knot_x = [0.0] + pits + [1.0]
        self._knot_y = [0.0] + [(k + 1) / (n + 1) for k in range(n)] + [1.0]
        self.n_fit = n
        return self

    def pit_cdf(self, z: float) -> float:
        """Empirical CDF G(z) of the fitted PIT values (linearly interpolated)."""
        xs, ys = self._knot_x, self._knot_y
        if not xs:
            raise ValueError("corrector is not fitted yet")
        if z <= 0.0:
            return 0.0
        if z >= 1.0:
            return 1.0
        lo, hi = 1, len(xs)  # first index with xs[lo] > z
        while lo < hi:
            mid = (lo + hi) // 2
            if xs[mid] <= z:
                lo = mid + 1
            else:
                hi = mid
        x0, x1 = xs[lo - 1], xs[lo]
        y0, y1 = ys[lo - 1], ys[lo]
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (z - x0) / (x1 - x0)

    def corrected_cdf(self, mu: float, sigma: float):
        """Return the corrected forecast CDF F_corrected(x) = G(F_raw(x))."""
        return lambda x: self.pit_cdf(gaussian_cdf(x, mu, sigma))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def pit_chi_square(pits: list[float], bins: int = 10) -> float:
    """Chi-square statistic of PIT values against the uniform ideal."""
    n = len(pits)
    counts = [0] * bins
    for z in pits:
        counts[min(int(z * bins), bins - 1)] += 1
    expected = n / bins
    return sum((c - expected) ** 2 / expected for c in counts)


def brier_score(pairs: list[tuple[float, int]]) -> float:
    """Mean squared error of forecast exceedance probabilities vs outcomes."""
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def run_demo(seed: int = 7, n_days: int = 1460, archive_days: int = 1095,
             threshold: float = 5.0) -> dict:
    """Fit PBC on the archive, score raw / corrected / climatology on holdout."""
    world = World.generate(n_days=n_days, seed=seed)
    model = BiasedModel()
    archive: list[tuple[float, float, float]] = []
    holdout: list[tuple[float, float, float]] = []
    for t in range(n_days):
        mu, sigma = model.forecast(t)
        entry = (mu, sigma, world.truth[t])
        (archive if t < archive_days else holdout).append(entry)

    corrector = PBCorrector().fit(archive)

    ys = [y for _, _, y in archive]
    climo_mu = sum(ys) / len(ys)
    climo_var = sum((v - climo_mu) ** 2 for v in ys) / len(ys)
    climo_sd = math.sqrt(climo_var)

    crps = {"raw": 0.0, "corrected": 0.0, "climatology": 0.0}
    brier = {"raw": [], "corrected": [], "climatology": []}
    pit_raw, pit_corr = [], []
    for mu, sigma, y in holdout:
        crps["raw"] += crps_gaussian_closed(mu, sigma, y)
        crps["climatology"] += crps_gaussian_closed(climo_mu, climo_sd, y)
        f_corr = corrector.corrected_cdf(mu, sigma)
        crps["corrected"] += crps_cdf(f_corr, y, mu - 8 * sigma, mu + 8 * sigma)
        pit_raw.append(gaussian_cdf(y, mu, sigma))
        pit_corr.append(f_corr(y))
        outcome = 1 if y > threshold else 0
        brier["raw"].append((gaussian_exceedance(threshold, mu, sigma), outcome))
        brier["corrected"].append((1.0 - f_corr(threshold), outcome))
        brier["climatology"].append(
            (gaussian_exceedance(threshold, climo_mu, climo_sd), outcome))

    m = len(holdout)
    return {
        "seed": seed,
        "n_archive": len(archive),
        "n_holdout": m,
        "threshold": threshold,
        "crps_raw": crps["raw"] / m,
        "crps_corrected": crps["corrected"] / m,
        "crps_climatology": crps["climatology"] / m,
        "brier_raw": brier_score(brier["raw"]),
        "brier_corrected": brier_score(brier["corrected"]),
        "brier_climatology": brier_score(brier["climatology"]),
        "pit_chi2_raw": pit_chi_square(pit_raw),
        "pit_chi2_corrected": pit_chi_square(pit_corr),
    }


def main() -> None:
    r = run_demo()
    line = "-" * 64
    print("Probabilistic Bias Correction (PBC) - toy demo scoreboard")
    print(f"archive: {r['n_archive']} days | holdout: {r['n_holdout']} days "
          f"| seed {r['seed']}")
    print(line)
    print(f"{'metric':<28}{'raw model':>12}{'PBC corrected':>14}{'climatology':>12}")
    print(line)
    print(f"{'CRPS (lower is better)':<28}{r['crps_raw']:>12.4f}"
          f"{r['crps_corrected']:>14.4f}{r['crps_climatology']:>12.4f}")
    print(f"{'Brier, P(y>%.0f) (lower)':<28}{r['brier_raw']:>12.4f}"
          f"{r['brier_corrected']:>14.4f}{r['brier_climatology']:>12.4f}"
          % r["threshold"])
    print(f"{'PIT chi-square, 10 bins':<28}{r['pit_chi2_raw']:>12.2f}"
          f"{r['pit_chi2_corrected']:>14.2f}{'':>12}")
    print(line)
    improv = 100.0 * (r["crps_raw"] - r["crps_corrected"]) / r["crps_raw"]
    print(f"PBC cuts CRPS by {improv:.1f}% vs the raw model and "
          f"{'beats' if r['crps_corrected'] < r['crps_climatology'] else 'trails'} "
          f"climatology on the holdout.")
    print("Toy synthetic data - illustrates the idea, not the paper's results.")


if __name__ == "__main__":
    main()
