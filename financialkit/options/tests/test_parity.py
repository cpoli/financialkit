"""Tests for put-call parity and Merton's no-arbitrage bounds."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_ATOL
from financialkit.options import black_scholes_price, forward_price, no_arbitrage_bounds, parity_residual, put_call_parity


def test_black_scholes_prices_satisfy_parity():
    K = np.linspace(60, 140, 9)
    c = black_scholes_price(100.0, K, 1.0, 0.05, 0.3, q=0.02)
    p = black_scholes_price(100.0, K, 1.0, 0.05, 0.3, q=0.02, option_type="put")
    np.testing.assert_allclose(parity_residual(c, p, 100.0, K, 1.0, 0.05, 0.02), 0.0, atol=DEFAULT_ATOL)
    np.testing.assert_allclose(put_call_parity(c, 100.0, K, 1.0, 0.05, 0.02), p, atol=DEFAULT_ATOL)
    np.testing.assert_allclose(put_call_parity(p, 100.0, K, 1.0, 0.05, 0.02, "put"), c, atol=DEFAULT_ATOL)


def test_mispriced_pair_has_the_arbitrage_profit_as_residual():
    c = black_scholes_price(50.0, 50.0, 0.5, 0.04, 0.25)
    p = black_scholes_price(50.0, 50.0, 0.5, 0.04, 0.25, option_type="put")
    assert parity_residual(c + 0.3, p, 50.0, 50.0, 0.5, 0.04) == pytest.approx(0.3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black_scholes_prices_lie_within_merton_bounds(option_type):
    K = np.linspace(50, 150, 11)
    for sigma in (0.01, 0.2, 3.0):
        price = black_scholes_price(100.0, K, 2.0, 0.05, sigma, q=0.01, option_type=option_type)
        lo, hi = no_arbitrage_bounds(100.0, K, 2.0, 0.05, 0.01, option_type)
        assert np.all(price >= lo - DEFAULT_ATOL) and np.all(price <= hi + DEFAULT_ATOL)


def test_bounds_are_attained_in_the_volatility_limits():
    lo, hi = no_arbitrage_bounds(100.0, 90.0, 1.0, 0.05)
    assert black_scholes_price(100.0, 90.0, 1.0, 0.05, 1e-6) == pytest.approx(lo)
    assert black_scholes_price(100.0, 90.0, 1.0, 0.05, 20.0) == pytest.approx(hi, rel=1e-6)


def test_forward_price_is_cost_of_carry_and_the_parity_forward():
    """F = S e^{(r - q) T}, and C - P = e^{-rT} (F - K)."""
    S, K, T, r, q = 100.0, np.array([80.0, 100.0, 120.0]), 2.0, 0.05, 0.02
    F = forward_price(S, T, r, q)
    assert F == pytest.approx(S * np.exp((r - q) * T), rel=1e-15)
    gap = black_scholes_price(S, K, T, r, 0.25, q) - black_scholes_price(S, K, T, r, 0.25, q, "put")
    np.testing.assert_allclose(gap, np.exp(-r * T) * (F - K), atol=DEFAULT_ATOL)
