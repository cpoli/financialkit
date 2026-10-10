"""Tests for financialkit.exotic_options.systems.variance_swaps."""

import numpy as np
import pytest

from financialkit.exotic_options import realized_variance, variance_swap_strike
from financialkit.integrators import gbm_paths
from financialkit.options import black_scholes_price


def _strip(vol, K, S=100.0, T=1.0, r=0.05, q=0.0):
    c, p = (black_scholes_price(S, K, T, r, vol(K), q, t) for t in ("call", "put"))
    return variance_swap_strike(K, c, p, S, T, r, q)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_flat_volatility_gives_sigma_squared(q):
    """Demeterfi et al. (1999): under Black-Scholes the fair variance is sigma^2.

    A 1-wide strike grid from 20 to 400 truncates and discretises the
    integral at the 1e-3 relative level."""
    res = _strip(lambda K: 0.2, np.arange(20.0, 400.0, 1.0), q=q)
    assert res.fair_variance == pytest.approx(0.04, rel=1e-3)
    assert res.K0 <= res.forward < res.K0 + 1.0
    assert res.contributions.sum() - (res.forward / res.K0 - 1) ** 2 / res.T == pytest.approx(res.fair_variance)


def test_skew_raises_the_fair_strike_above_at_the_money_vol():
    """A downward skew puts more weight on cheap-strike puts, so K_var exceeds the ATM-forward volatility squared."""
    res = _strip(lambda K: 0.2 - 0.1 * np.log(K / 105.127), np.arange(10.0, 600.0, 0.5))
    assert res.vol_strike > 0.2 + 0.003


def test_realized_variance_of_gbm():
    """E[sum of squared log returns] / T = sigma^2 + (r - sigma^2/2)^2 T / n."""
    sigma, r, T, n = 0.3, 0.05, 1.0, 252
    _, paths = gbm_paths(100.0, r, sigma, T, n, 4000, rng=0)
    rv = realized_variance(paths, T)
    expected = sigma**2 + (r - 0.5 * sigma**2) ** 2 * T / n
    assert abs(rv.mean() - expected) < 4 * rv.std() / np.sqrt(rv.size)


def test_bad_grid():
    with pytest.raises(ValueError):
        variance_swap_strike([100.0, 90.0, 110.0], [1, 2, 3], [1, 2, 3], 100.0, 1.0, 0.0)
