"""Tests for implied volatility, the Brenner-Subrahmanyam approximation and the smile."""

import numpy as np
import pytest

from financialkit.options import black76_price, black_scholes_price, brenner_subrahmanyam_vol, implied_vol_smile, implied_volatility


def test_hull_section_15_11():
    """Hull, 10th ed., Section 15.11: a 1.875 call (S = 21, K = 20, r = 10%, 3 months) implies 23.5%."""
    assert implied_volatility(1.875, 21.0, 20.0, 0.25, 0.10) == pytest.approx(0.235, abs=5e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_round_trip_across_strikes_and_vols(option_type):
    K = np.array([[70.0], [100.0], [140.0]])
    sigma = np.array([[0.1, 0.3, 1.2]])
    prices = black_scholes_price(100.0, K, 1.5, 0.04, sigma, 0.02, option_type)
    np.testing.assert_allclose(implied_volatility(prices, 100.0, K, 1.5, 0.04, 0.02, option_type), np.broadcast_to(sigma, (3, 3)), atol=1e-7)


def test_black76_implied_vol_via_forward_and_q_equals_r():
    price = black76_price(20.0, 22.0, 0.5, 0.06, 0.3)
    assert implied_volatility(price, 20.0, 22.0, 0.5, 0.06, q=0.06) == pytest.approx(0.3, abs=1e-8)


@pytest.mark.parametrize("price", [0.0, 2.0, 100.0, 150.0])
def test_prices_outside_merton_bounds_raise(price):
    # lower bound of this call is 100 - 98 e^{-0.05} ~ 6.78, upper bound 100
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(price, 100.0, 98.0, 1.0, 0.05)


def test_brenner_subrahmanyam_error_is_third_order_in_vol():
    for sigma, T in [(0.1, 0.25), (0.2, 1.0), (0.4, 1.0)]:
        F = 100.0 * np.exp(0.03 * T)
        c = black_scholes_price(100.0, F, T, 0.03, sigma)
        approx = brenner_subrahmanyam_vol(c, 100.0, T, 0.03)
        # 2N(x/2) - 1 = x/sqrt(2 pi) (1 - x^2/24 + ...), so the relative error is ~ sigma^2 T / 24
        assert approx / sigma - 1 == pytest.approx(-(sigma**2) * T / 24, rel=0.1, abs=1e-6)


def test_smile_from_skewed_prices_slopes_down():
    K = np.linspace(70, 130, 13)
    # a mixture of two lognormals with the same forward: a standard arbitrage-free skew
    w, s1, s2 = 0.3, 0.45, 0.15
    prices = w * black_scholes_price(100.0, K, 0.5, 0.0, s1) + (1 - w) * black_scholes_price(100.0, K, 0.5, 0.0, s2)
    smile = implied_vol_smile(prices, K, 100.0, 0.5, 0.0)
    assert smile.vols.shape == K.shape and smile.forward == pytest.approx(100.0)
    # a symmetric mixture gives a smile: the wings exceed the at-the-money vol
    atm = smile.vols[K == 100.0][0]
    assert smile.vols[0] > atm and smile.vols[-1] > atm
    np.testing.assert_allclose(smile.log_moneyness, np.log(K / 100.0))
