"""Tests for Heston (1993) and the Hull-White (1987) mixing formula."""

import numpy as np
import pytest

from financialkit.options import black_scholes_price
from financialkit.stochastic_volatility import HestonModel, heston_price, hull_white_mixing_mc, hull_white_price


def test_andersen_case_i_and_ii():
    """Andersen (2008), Section 7 test cases: Case I (kappa = 0.5, xi = 1, rho = -0.9, T = 10) gives
    13.0847 and Case II (kappa = 0.3, xi = 0.9, rho = -0.5, T = 15) gives 16.649 for the at-the-money call."""
    case1 = HestonModel(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9)
    case2 = HestonModel(v0=0.04, kappa=0.3, theta=0.04, xi=0.9, rho=-0.5)
    assert heston_price(case1, 100.0, 100.0, 10.0, 0.0) == pytest.approx(13.0847, abs=5e-5)
    assert heston_price(case2, 100.0, 100.0, 15.0, 0.0) == pytest.approx(16.649, abs=5e-4)
    assert case1.price(100.0, 100.0, 10.0, 0.0) == pytest.approx(13.0847, abs=5e-5)


def test_heston_without_vol_of_vol_noise_is_black_scholes():
    """With v0 = theta the variance is deterministic as xi -> 0, so the price tends to Black-Scholes."""
    m = HestonModel(0.04, 2.0, 0.04, 1e-4, 0.0)
    K = np.array([80.0, 100.0, 125.0])
    assert np.asarray(m.price(100.0, K, 1.0, 0.03)) == pytest.approx(black_scholes_price(100.0, K, 1.0, 0.03, 0.2), abs=1e-6)


def test_negative_correlation_gives_a_downward_skew():
    K = np.array([80.0, 100.0, 120.0])
    down = np.asarray(HestonModel(0.04, 1.5, 0.04, 0.5, -0.7).implied_vol(100.0, K, 1.0, 0.0))
    up = np.asarray(HestonModel(0.04, 1.5, 0.04, 0.5, 0.7).implied_vol(100.0, K, 1.0, 0.0))
    assert down[0] > down[1] > down[2]
    assert up[0] < up[1] < up[2]


def test_feller_and_variance_mean():
    m = HestonModel(0.09, 2.0, 0.04, 0.3, -0.5)
    assert m.feller_satisfied
    assert not HestonModel(0.04, 0.5, 0.04, 1.0, -0.9).feller_satisfied
    assert m.variance_mean(0.0) == pytest.approx(0.09)
    assert m.variance_mean(50.0) == pytest.approx(0.04)


def test_heston_rejects_bad_parameters():
    with pytest.raises(ValueError):
        HestonModel(0.04, 1.0, 0.04, 0.3, -1.0)
    with pytest.raises(ValueError):
        HestonModel(0.04, -1.0, 0.04, 0.3, 0.0)


def test_hull_white_reduces_to_black_scholes():
    assert hull_white_price(100.0, 110.0, 0.5, 0.05, 0.25, 0.0, option_type="put") == black_scholes_price(100.0, 110.0, 0.5, 0.05, 0.25, option_type="put")


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_hull_white_series_matches_mixing_monte_carlo(K):
    """For xi^2 T = 0.09 the series truncation error is far below the Monte Carlo error."""
    res = hull_white_mixing_mc(100.0, K, 1.0, 0.0, 0.2, 0.3, n_paths=40_000, rng=11)
    assert abs(res.estimate - hull_white_price(100.0, K, 1.0, 0.0, 0.2, 0.3)) < 4 * res.std_error


def test_hull_white_smile_is_symmetric_and_atm_price_falls():
    """Uncorrelated stochastic variance lowers the at-the-money price (the BS price is concave in variance there)
    and gives a smile symmetric in log-moneyness."""
    from financialkit.options import implied_volatility

    F = 100.0
    K = F * np.exp(np.array([-0.2, 0.0, 0.2]))
    prices = np.asarray(hull_white_price(100.0, K, 1.0, 0.0, 0.2, 0.3))
    assert prices[1] < black_scholes_price(100.0, 100.0, 1.0, 0.0, 0.2)
    vols = np.asarray(implied_volatility(prices, 100.0, K, 1.0, 0.0))
    assert vols[0] == pytest.approx(vols[2], abs=1e-6)
    assert vols[0] > vols[1]


def test_hull_white_price_is_smooth_in_xi_across_the_series_switch():
    """(price - BS) / xi^2 tends to a constant as xi -> 0. The closed-form moments lose it to
    cancellation for small xi^2 T, so they hand over to their Taylor series below 0.1."""
    args = (100.0, 100.0, 1.0, 0.05, 0.2)
    bs = black_scholes_price(*args)
    xi = np.array([0.005, 0.0099, 0.0101, 0.02])
    ratio = np.array([(hull_white_price(*args, x) - bs) / x**2 for x in xi])
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-3)
    edge = np.sqrt(0.1)
    assert hull_white_price(*args, edge * (1 - 1e-9)) == pytest.approx(hull_white_price(*args, edge * (1 + 1e-9)), rel=1e-10)
