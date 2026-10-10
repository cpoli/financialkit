"""Tests for the closed-form pricers: Bachelier, Black-Scholes-Merton, Black-76, Garman-Kohlhagen."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import lognorm

from financialkit.constants import DEFAULT_ATOL, DEFAULT_RTOL
from financialkit.options import bachelier_price, black76_price, black_scholes_price, garman_kohlhagen_price


def test_black_scholes_hull_example_15_6():
    """Hull, 10th ed., Example 15.6: c = 4.76, p = 0.81."""
    assert black_scholes_price(42.0, 40.0, 0.5, 0.10, 0.20) == pytest.approx(4.76, abs=5e-3)
    assert black_scholes_price(42.0, 40.0, 0.5, 0.10, 0.20, option_type="put") == pytest.approx(0.81, abs=5e-3)


def test_merton_dividend_yield_hull_example_17_1():
    """Hull, 10th ed., Example 17.1: index call with q = 3% is worth 51.83."""
    assert black_scholes_price(930.0, 900.0, 2 / 12, 0.08, 0.20, q=0.03) == pytest.approx(51.83, abs=5e-3)


def test_black_scholes_is_discounted_lognormal_expectation():
    S, K, T, r, q, sigma = 100.0, 95.0, 0.75, 0.04, 0.01, 0.3
    dist = lognorm(s=sigma * np.sqrt(T), scale=S * np.exp((r - q - 0.5 * sigma**2) * T))
    expected = np.exp(-r * T) * quad(lambda s: (s - K) * dist.pdf(s), K, np.inf)[0]
    assert black_scholes_price(S, K, T, r, sigma, q) == pytest.approx(expected, rel=1e-7)


def test_black_scholes_broadcasts_over_strikes_and_maturities():
    K = np.linspace(80, 120, 5)[:, None]
    T = np.array([0.25, 1.0, 2.0])[None, :]
    prices = black_scholes_price(100.0, K, T, 0.03, 0.2)
    assert prices.shape == (5, 3)
    assert prices[0, 0] == pytest.approx(black_scholes_price(100.0, 80.0, 0.25, 0.03, 0.2))


def test_zero_volatility_and_expiry_give_intrinsic_values():
    assert black_scholes_price(110.0, 100.0, 0.0, 0.05, 0.2) == pytest.approx(10.0)
    assert black_scholes_price(90.0, 100.0, 1.0, 0.0, 0.0, option_type="put") == pytest.approx(10.0)


def test_black76_hull_example_18_7():
    """Hull, 10th ed., Example 18.7: European futures put worth 1.12."""
    assert black76_price(20.0, 20.0, 4 / 12, 0.09, 0.25, "put") == pytest.approx(1.12, abs=5e-3)


def test_black76_equals_black_scholes_on_the_forward():
    S, K, T, r, q, sigma = 100.0, 105.0, 1.5, 0.04, 0.02, 0.25
    F = S * np.exp((r - q) * T)
    assert black76_price(F, K, T, r, sigma) == pytest.approx(black_scholes_price(S, K, T, r, sigma, q), rel=DEFAULT_RTOL)


def test_garman_kohlhagen_haug_example():
    """Haug, 2nd ed., Section 1.1.6: currency call worth 0.0291."""
    assert garman_kohlhagen_price(1.56, 1.60, 0.5, 0.06, 0.08, 0.12) == pytest.approx(0.0291, abs=5e-5)


def test_bachelier_at_the_money_and_parity():
    S, T, r, sn = 100.0, 2.0, 0.03, 15.0
    assert bachelier_price(S, S * np.exp(r * T), T, r, sn) == pytest.approx(np.exp(-r * T) * sn * np.sqrt(T / (2 * np.pi)), rel=DEFAULT_RTOL)
    K = np.array([80.0, 100.0, 130.0])
    c, p = bachelier_price(S, K, T, r, sn), bachelier_price(S, K, T, r, sn, option_type="put")
    np.testing.assert_allclose(c - p, S - K * np.exp(-r * T), atol=DEFAULT_ATOL)


def test_bachelier_close_to_black_scholes_with_matched_volatility():
    """For short maturities near the money, sigma_N ~= sigma * S."""
    bs = black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.2)
    assert bachelier_price(100.0, 100.0, 0.05, 0.0, 20.0) == pytest.approx(bs, rel=2e-3)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="straddle")
