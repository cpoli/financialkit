"""Tests for financialkit.exotic_options.systems.asian."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.exotic_options import arithmetic_asian_mc, geometric_asian_price, turnbull_wakeman_price
from financialkit.integrators import gbm_paths


def test_kemna_vorst_haug_example():
    """Haug (2007), Section 4.20.1: geometric average-rate put, 4.6922."""
    assert geometric_asian_price(80.0, 85.0, 0.25, 0.05, 0.20, q=-0.03, option_type="put") == pytest.approx(4.6922, abs=1e-4)


def test_discrete_geometric_tends_to_continuous():
    args = (100.0, np.array([90.0, 100.0, 110.0]), 1.0, 0.05, 0.3, 0.01)
    gap = [np.max(np.abs(geometric_asian_price(*args, n_fixings=n) - geometric_asian_price(*args))) for n in (10, 100, 1000)]
    assert gap[0] > gap[1] > gap[2] and gap[2] < 1e-2


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_discrete_geometric_against_monte_carlo(option_type):
    S, K, T, r, sigma, q, n = 100.0, 105.0, 1.0, 0.05, 0.25, 0.02, 12
    _, paths = gbm_paths(S, r - q, sigma, T, n, 200_000, rng=3)
    G = np.exp(np.log(paths[:, 1:]).mean(axis=1))
    y = np.exp(-r * T) * np.maximum((1 if option_type == "call" else -1) * (G - K), 0.0)
    exact = geometric_asian_price(S, K, T, r, sigma, q, option_type, n_fixings=n)
    assert abs(y.mean() - exact) < 4 * y.std() / np.sqrt(y.size)


def test_arithmetic_control_variate_agrees_with_crude_and_reduces_variance():
    res = arithmetic_asian_mc(100.0, 100.0, 1.0, 0.05, 0.2, n_fixings=12, n_paths=50_000, rng=4)
    crude = arithmetic_asian_mc(100.0, 100.0, 1.0, 0.05, 0.2, n_fixings=12, n_paths=50_000, control_variate=False, rng=5)
    assert res.r_squared > 0.99
    assert res.std_error < crude.std_error / 10
    assert abs(res.estimate - crude.estimate) < 4 * np.hypot(res.std_error, crude.std_error)


def test_turnbull_wakeman_close_to_monte_carlo():
    """Moment matching is an approximation: within 0.5% of the controlled MC price at 20% volatility."""
    for K in (90.0, 100.0, 110.0):
        mc = arithmetic_asian_mc(100.0, K, 1.0, 0.05, 0.2, n_fixings=12, n_paths=50_000, rng=6)
        tw = turnbull_wakeman_price(100.0, K, 1.0, 0.05, 0.2, n_fixings=12)
        assert abs(tw - mc.estimate) < 0.005 * mc.estimate + 4 * mc.std_error


@pytest.mark.parametrize("q", [0.0, 0.05, 0.09, 0.2])
def test_turnbull_wakeman_continuous_matches_dense_fixings(q):
    """q = 0.05 makes b = 0 and q = 0.09 makes b + sigma^2 = 0, the formula's removable singularities."""
    cont = turnbull_wakeman_price(100.0, 100.0, 1.0, 0.05, 0.2, q=q)
    assert cont == pytest.approx(turnbull_wakeman_price(100.0, 100.0, 1.0, 0.05, 0.2, q=q, n_fixings=20_000), rel=1e-4)


def test_turnbull_wakeman_first_moment_is_exact():
    """A zero-strike call is the discounted mean of the average, whatever the second moment."""
    for n in (None, 4):
        mean = np.expm1(0.03) / 0.03 if n is None else np.mean(np.exp(0.03 * np.arange(1, 5) / 4))
        assert turnbull_wakeman_price(100.0, 1e-12, 1.0, 0.05, 0.3, q=0.02, n_fixings=n) == pytest.approx(100.0 * np.exp(-0.05) * mean, rel=DEFAULT_RTOL)
