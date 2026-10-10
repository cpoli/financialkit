"""Tests for financialkit.exotic_options.systems.lookbacks."""

import numpy as np
import pytest

from financialkit.exotic_options import fixed_lookback_price, floating_lookback_price, lookback_option_mc


def test_haug_lookback_examples():
    """Haug (2007), Section 4.15.1 (floating call 25.3533) and Table 4-12 (fixed, sigma = 10%: call 13.2687, put 0.6899)."""
    assert floating_lookback_price(120.0, 0.5, 0.10, 0.30, q=0.06, S_extreme=100.0) == pytest.approx(25.3533, abs=1e-4)
    assert fixed_lookback_price(100.0, 95.0, 0.5, 0.1, 0.1) == pytest.approx(13.2687, abs=1e-4)
    assert fixed_lookback_price(100.0, 95.0, 0.5, 0.1, 0.1, option_type="put") == pytest.approx(0.6899, abs=1e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [None, 95.0, 110.0])
@pytest.mark.parametrize("q", [0.0, 0.05])
def test_bridge_monte_carlo_matches_closed_forms(option_type, K, q):
    """q = 0.05 = r exercises the zero-carry limit."""
    exact = floating_lookback_price(100.0, 1.0, 0.05, 0.25, q, option_type) if K is None else fixed_lookback_price(100.0, K, 1.0, 0.05, 0.25, q, option_type)
    res = lookback_option_mc(100.0, 1.0, 0.05, 0.25, q, option_type, K=K, n_steps=10, n_paths=60_000, rng=11)
    assert abs(res.estimate - exact) < 4 * res.std_error


def test_zero_carry_limit_is_continuous():
    f = [floating_lookback_price(100.0, 1.0, 0.05, 0.3, q) for q in (0.05 - 1e-4, 0.05, 0.05 + 1e-4)]
    assert abs(f[1] - 0.5 * (f[0] + f[2])) < 1e-6
    g = [fixed_lookback_price(100.0, 105.0, 1.0, 0.05, 0.3, q, "put", S_extreme=95.0) for q in (0.05 - 1e-4, 0.05, 0.05 + 1e-4)]
    assert abs(g[1] - 0.5 * (g[0] + g[2])) < 1e-6


def test_floating_put_is_fixed_call_struck_at_the_maximum():
    """max(M, S_max) - S_T = (max(M, S_max) - S_max) + S_max - S_T."""
    S, Smax, T, r, sigma, q = 100.0, 110.0, 0.75, 0.04, 0.3, 0.01
    lhs = floating_lookback_price(S, T, r, sigma, q, "put", S_extreme=Smax)
    rhs = fixed_lookback_price(S, Smax, T, r, sigma, q, "call", S_extreme=Smax) + Smax * np.exp(-r * T) - S * np.exp(-q * T)
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_in_the_money_fixed_call_locks_in_intrinsic():
    S, T, r, sigma = 100.0, 0.5, 0.05, 0.2
    deep = fixed_lookback_price(S, 80.0, T, r, sigma, S_extreme=120.0)
    assert deep > np.exp(-r * T) * 40.0
    assert deep - np.exp(-r * T) * 40.0 == pytest.approx(fixed_lookback_price(S, 120.0, T, r, sigma, S_extreme=120.0), rel=1e-12)


def test_discrete_monitoring_is_cheaper():
    cont = floating_lookback_price(100.0, 1.0, 0.05, 0.3)
    disc = lookback_option_mc(100.0, 1.0, 0.05, 0.3, n_steps=12, n_paths=50_000, continuous=False, rng=12)
    assert disc.estimate + 4 * disc.std_error < cont
