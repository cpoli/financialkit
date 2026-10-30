"""Tests for financialkit.monte_carlo.systems.basics and the result container."""

import numpy as np
import pytest
from scipy.stats import kstest

from financialkit.monte_carlo import MonteCarloResult, box_muller, box_muller_normals, european_option_mc, monte_carlo_integrate, sample_mean, vanilla_payoff
from financialkit.options import black_scholes_price


def test_integrate_pi_within_standard_errors():
    res = monte_carlo_integrate(lambda u: 4.0 * (np.sum(u**2, axis=1) < 1), 200_000, dim=2, rng=1)
    assert abs(res.estimate - np.pi) < 4 * res.std_error


def test_standard_error_falls_like_inverse_sqrt_n():
    se = [monte_carlo_integrate(lambda u: u[:, 0] ** 2, n, rng=2).std_error for n in (1_000, 100_000)]
    # Var(U^2) = 4/45, so se = sqrt(4/45/n); a 100x sample cuts it 10x.
    assert se[0] / se[1] == pytest.approx(10.0, rel=0.1)
    assert se[1] == pytest.approx(np.sqrt(4 / 45 / 100_000), rel=0.05)


def test_confidence_interval_coverage():
    """95% intervals cover the true mean about 95% of the time (binomial 4-sigma band)."""
    hits = sum(monte_carlo_integrate(lambda u: u[:, 0], 200, rng=s).covers(0.5) for s in range(400))
    assert abs(hits / 400 - 0.95) < 4 * np.sqrt(0.95 * 0.05 / 400)


def test_result_properties():
    res = MonteCarloResult(1.0, 0.1, 100)
    assert res.half_width == pytest.approx(0.196, abs=1e-3)
    assert res.ci == pytest.approx((1.0 - res.half_width, 1.0 + res.half_width))
    assert res.variance == pytest.approx(1.0)
    assert MonteCarloResult(1.0, 0.05, 100).variance_ratio(res) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        sample_mean([1.0])


def test_box_muller_is_normal_and_independent():
    z = box_muller_normals(100_001, rng=3)
    assert z.size == 100_001
    assert kstest(z, "norm").pvalue > 1e-3
    z1, z2 = box_muller(*np.random.default_rng(4).random((2, 100_000)))
    assert abs(np.corrcoef(z1, z2)[0, 1]) < 4 / np.sqrt(100_000)


def test_box_muller_known_point():
    z1, z2 = box_muller(np.exp(-0.5), 0.0)
    assert (float(z1), float(z2)) == pytest.approx((1.0, 0.0))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_boyle_european_matches_black_scholes(option_type):
    """Hull, 10th ed., Example 15.6: c = 4.76, p = 0.81."""
    exact = black_scholes_price(42.0, 40.0, 0.5, 0.10, 0.20, option_type=option_type)
    res = european_option_mc(42.0, 40.0, 0.5, 0.10, 0.20, option_type=option_type, n_paths=200_000, rng=5)
    assert abs(res.estimate - exact) < 4 * res.std_error


def test_european_with_dividends_and_given_normals():
    z = np.random.default_rng(6).standard_normal(200_000)
    exact = black_scholes_price(100.0, 95.0, 1.0, 0.03, 0.25, q=0.02)
    res = european_option_mc(100.0, 95.0, 1.0, 0.03, 0.25, q=0.02, z=z)
    assert res.n_samples == z.size
    assert abs(res.estimate - exact) < 4 * res.std_error


def test_vanilla_payoff_rejects_unknown_option_type():
    with pytest.raises(ValueError, match="option_type"):
        vanilla_payoff(np.array([90.0, 110.0]), 100.0, "straddle")
