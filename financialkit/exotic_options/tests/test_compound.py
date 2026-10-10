"""Tests for financialkit.exotic_options.systems.compound."""

import numpy as np
import pytest

from financialkit.exotic_options import chooser_price, compound_option_price, critical_price, forward_start_price
from financialkit.options import black_scholes_price

S, K1, K2, T1, T2, r, sigma, q = 100.0, 4.0, 100.0, 0.5, 1.0, 0.05, 0.25, 0.02


def test_geske_haug_example():
    """Haug (2007), Section 4.6: put on call, 21.1965.

    Haug's value comes from a Drezner-type approximation of the bivariate
    normal, good to a few 1e-5 in probability; Owen's T is exact, so allow
    5e-4 on a price of 21."""
    assert compound_option_price(500.0, 50.0, 520.0, 0.25, 0.5, 0.08, 0.35, q=0.03, option_type="put") == pytest.approx(21.1965, abs=5e-4)


@pytest.mark.parametrize("underlying", ["call", "put"])
def test_compound_put_call_parity(underlying):
    """max(V - K1, 0) - max(K1 - V, 0) = V - K1."""
    c = compound_option_price(S, K1, K2, T1, T2, r, sigma, q, "call", underlying)
    p = compound_option_price(S, K1, K2, T1, T2, r, sigma, q, "put", underlying)
    assert c - p == pytest.approx(black_scholes_price(S, K2, T2, r, sigma, q, underlying) - K1 * np.exp(-r * T1), abs=1e-10)


@pytest.mark.parametrize("underlying", ["call", "put"])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_compound_against_monte_carlo(option_type, underlying):
    z = np.random.default_rng(13).standard_normal(200_000)
    S1 = S * np.exp((r - q - 0.5 * sigma**2) * T1 + sigma * np.sqrt(T1) * z)
    V = black_scholes_price(S1, K2, T2 - T1, r, sigma, q, underlying)
    y = np.exp(-r * T1) * np.maximum((1 if option_type == "call" else -1) * (V - K1), 0.0)
    exact = compound_option_price(S, K1, K2, T1, T2, r, sigma, q, option_type, underlying)
    assert abs(y.mean() - exact) < 4 * y.std() / np.sqrt(y.size)


def test_critical_price_and_validation():
    for u in ("call", "put"):
        I = critical_price(K1, K2, T2 - T1, r, sigma, q, u)
        assert black_scholes_price(I, K2, T2 - T1, r, sigma, q, u) == pytest.approx(K1, abs=1e-10)
    with pytest.raises(ValueError):
        critical_price(200.0, 100.0, 0.5, 0.05, 0.2, underlying_type="put")
    with pytest.raises(ValueError):
        compound_option_price(S, K1, K2, 1.0, 0.5, r, sigma)


def test_forward_start_haug_example_and_monte_carlo():
    """Haug (2007), Section 4.3: forward-start call, 4.4064."""
    assert forward_start_price(60.0, 1.1, 0.25, 1.0, 0.08, 0.30, q=0.04) == pytest.approx(4.4064, abs=1e-4)
    t, T, alpha = 0.4, 1.0, 0.95
    z = np.random.default_rng(14).standard_normal((2, 200_000))
    St = S * np.exp((r - q - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z[0])
    ST = St * np.exp((r - q - 0.5 * sigma**2) * (T - t) + sigma * np.sqrt(T - t) * z[1])
    y = np.exp(-r * T) * np.maximum(alpha * St - ST, 0.0)
    exact = forward_start_price(S, alpha, t, T, r, sigma, q, "put")
    assert abs(y.mean() - exact) < 4 * y.std() / np.sqrt(y.size)


def test_forward_start_starting_now_is_vanilla():
    assert forward_start_price(S, 1.0, 0.0, 1.0, r, sigma, q) == pytest.approx(black_scholes_price(S, S, 1.0, r, sigma, q))


def test_chooser_haug_example_and_limits():
    """Haug (2007), Section 4.4.1: simple chooser, 6.1071."""
    assert chooser_price(50.0, 50.0, 0.25, 0.5, 0.08, 0.25) == pytest.approx(6.1071, abs=1e-4)
    c = black_scholes_price(S, 105.0, 1.0, r, sigma, q)
    p = black_scholes_price(S, 105.0, 1.0, r, sigma, q, "put")
    assert chooser_price(S, 105.0, 0.0, 1.0, r, sigma, q) == pytest.approx(max(c, p), rel=1e-12)
    assert chooser_price(S, 105.0, 1.0, 1.0, r, sigma, q) == pytest.approx(c + p, rel=1e-12)
    mid = chooser_price(S, 105.0, 0.5, 1.0, r, sigma, q)
    assert max(c, p) < mid < c + p
