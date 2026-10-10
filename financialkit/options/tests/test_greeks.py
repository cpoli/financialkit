"""Tests for the analytic Black-Scholes-Merton Greeks."""

import numpy as np
import pytest

from financialkit.options import black_scholes_greeks, black_scholes_price, finite_difference_greeks

FIELDS = ("delta", "gamma", "vega", "theta", "rho", "epsilon", "vanna", "volga")


def test_hull_example_19_1():
    """Hull, 10th ed., Examples 19.1 and Sections 19.4-19.8: delta 0.522, gamma 0.066,
    vega 12.1, theta -4.31 per year, rho 8.91."""
    g = black_scholes_greeks(49.0, 50.0, 0.3846, 0.05, 0.20)
    assert g.delta == pytest.approx(0.522, abs=5e-4)
    assert g.gamma == pytest.approx(0.066, abs=5e-4)
    assert g.vega == pytest.approx(12.1, abs=0.05)
    assert g.theta == pytest.approx(-4.31, abs=5e-3)
    assert g.rho == pytest.approx(8.91, abs=5e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("S,K,T,q", [(100.0, 100.0, 1.0, 0.0), (80.0, 100.0, 0.5, 0.03), (130.0, 100.0, 2.0, 0.01)])
def test_greeks_match_finite_differences(option_type, S, K, T, q):
    g = black_scholes_greeks(S, K, T, 0.05, 0.25, q, option_type)
    fd = finite_difference_greeks(black_scholes_price, S, K, T, 0.05, 0.25, q, option_type)
    for name in FIELDS:
        # second differences with h = 1e-4 lose about half the digits
        assert getattr(g, name) == pytest.approx(getattr(fd, name), rel=1e-5, abs=1e-5), name


def test_greeks_satisfy_the_black_scholes_pde():
    S, K, T, r, q, sigma = np.linspace(60, 140, 9), 100.0, 0.8, 0.04, 0.015, 0.3
    for option_type in ("call", "put"):
        g = black_scholes_greeks(S, K, T, r, sigma, q, option_type)
        V = black_scholes_price(S, K, T, r, sigma, q, option_type)
        lhs = g.theta + 0.5 * sigma**2 * S**2 * g.gamma + (r - q) * S * g.delta
        np.testing.assert_allclose(lhs, r * V, atol=1e-10)


def test_call_put_delta_gap_is_the_discounted_dividend_factor():
    c = black_scholes_greeks(100.0, 95.0, 1.0, 0.05, 0.2, 0.03)
    p = black_scholes_greeks(100.0, 95.0, 1.0, 0.05, 0.2, 0.03, "put")
    assert c.delta - p.delta == pytest.approx(np.exp(-0.03))
    assert c.gamma == pytest.approx(p.gamma) and c.vega == pytest.approx(p.vega)
