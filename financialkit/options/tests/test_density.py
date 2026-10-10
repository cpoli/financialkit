"""Tests for the Breeden-Litzenberger risk-neutral density."""

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import lognorm

from financialkit.options import black_scholes_price, breeden_litzenberger_density


def test_recovers_lognormal_density_on_nonuniform_grid():
    K = 100.0 * np.exp(np.linspace(-1.0, 1.0, 401))  # smoothly graded spacing, finer below the forward
    res = breeden_litzenberger_density(K, black_scholes_price(100.0, K, 0.5, 0.04, 0.3, q=0.01), 0.5, 0.04)
    exact = lognorm.pdf(res.strikes, s=0.3 * np.sqrt(0.5), scale=100 * np.exp((0.04 - 0.01 - 0.045) * 0.5))
    # second difference error is O(h^2 f'') with h <= 1.4; graded spacing keeps it second order
    np.testing.assert_allclose(res.density, exact, atol=2e-5)


def test_density_integrates_to_one_and_has_the_forward_as_mean():
    K = np.linspace(1.0, 400.0, 4000)
    res = breeden_litzenberger_density(K, black_scholes_price(100.0, K, 1.0, 0.05, 0.25), 1.0, 0.05)
    assert trapezoid(res.density, res.strikes) == pytest.approx(1.0, abs=1e-4)
    assert trapezoid(res.strikes * res.density, res.strikes) == pytest.approx(100 * np.exp(0.05), rel=1e-4)


def test_rejects_bad_grids():
    with pytest.raises(ValueError):
        breeden_litzenberger_density([1.0, 2.0], [1.0, 0.5], 1.0, 0.0)
    with pytest.raises(ValueError):
        breeden_litzenberger_density([1.0, 3.0, 2.0], [1.0, 0.5, 0.2], 1.0, 0.0)
