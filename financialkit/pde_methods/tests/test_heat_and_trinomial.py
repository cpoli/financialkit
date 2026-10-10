"""Tests for the heat-equation transform and the explicit log-spot scheme as a trinomial tree."""

import numpy as np
import pytest

from financialkit.options import CRRTree, black_scholes_greeks, black_scholes_price
from financialkit.pde_methods import explicit_trinomial_probabilities, heat_equation_exponents, heat_equation_price, log_explicit_price


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("S,q", [(100.0, 0.0), (90.0, 0.03), (120.0, 0.06)])
def test_heat_equation_price_matches_black_scholes(S, q, option_type):
    exact = black_scholes_price(S, 100.0, 1.0, 0.05, 0.25, q, option_type)
    res = heat_equation_price(S, 100.0, 1.0, 0.05, 0.25, q, option_type)
    # O(dx^2 + dtau^2) error with 400 x-intervals
    assert res.price == pytest.approx(exact, abs=5e-3)
    assert res.delta == pytest.approx(black_scholes_greeks(S, 100.0, 1.0, 0.05, 0.25, q, option_type).delta, abs=2e-3)


def test_heat_transform_reproduces_the_payoff_and_solves_the_heat_equation():
    res = heat_equation_price(100.0, 100.0, 1.0, 0.05, 0.25, n_time=200)
    np.testing.assert_allclose(res.surface[-1], np.maximum(res.S - 100.0, 0.0), atol=1e-10)
    assert res.tau[-1] == pytest.approx(0.5 * 0.25**2)
    # the exact heat kernel solution and the PDE solution agree in u-space too
    alpha, beta = heat_equation_exponents(0.05, 0.25)
    u_exact = black_scholes_price(res.S, 100.0, 1.0, 0.05, 0.25) / 100.0 * np.exp(-alpha * res.x - beta * res.tau[-1])
    centre = np.abs(res.x) < 0.5
    np.testing.assert_allclose(res.u[-1][centre], u_exact[centre], atol=1e-4)


def test_heat_equation_exponents():
    alpha, beta = heat_equation_exponents(0.05, 0.25, 0.01)
    kq, k = 2 * 0.04 / 0.0625, 2 * 0.05 / 0.0625
    assert alpha == pytest.approx(-(kq - 1) / 2) and beta == pytest.approx(-((kq - 1) ** 2) / 4 - k)


def test_trinomial_probabilities_match_log_price_moments():
    sigma, r, q, dt, dx = 0.3, 0.05, 0.01, 0.01, 0.05
    p = explicit_trinomial_probabilities(sigma, r, dt, dx, q)
    nu = r - q - 0.5 * sigma**2
    assert p.total == pytest.approx(1.0)
    assert (p.p_up - p.p_down) * dx == pytest.approx(nu * dt)  # mean of the log increment
    assert (p.p_up + p.p_down) * dx**2 == pytest.approx(sigma**2 * dt)  # its second moment
    assert p.discount == pytest.approx(1 / (1 + r * dt))
    assert not explicit_trinomial_probabilities(sigma, r, dt, 0.02, q).is_probability  # p_mid < 0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_log_explicit_price_matches_black_scholes(option_type):
    exact = black_scholes_price(100.0, 105.0, 1.0, 0.05, 0.2, 0.02, option_type)
    res = log_explicit_price(100.0, 105.0, 1.0, 0.05, 0.2, 0.02, option_type, n_time=800)
    assert res.stability.number == pytest.approx(1 / 3)
    # trinomial-tree error O(1/n) with an oscillating constant
    assert res.price == pytest.approx(exact, abs=0.01)


def test_log_explicit_american_put_matches_crr():
    """Hull, 10th ed., Example 21.1: the American put worth 4.28 by a fine tree."""
    reference = CRRTree(5000).price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True).price
    res = log_explicit_price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True, n_time=1000)
    assert res.price == pytest.approx(reference, abs=5e-3)


def test_value_at_spot_ignores_the_boundaries():
    # node values within the tree's cone never feel the Dirichlet data
    a = log_explicit_price(100.0, 100.0, 1.0, 0.05, 0.2, n_time=100)
    b = log_explicit_price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="put", n_time=100)
    parity = 100.0 - 100.0 * np.exp(-0.05)
    assert a.price - b.price == pytest.approx(parity, abs=5e-3)
    assert a.S[100] == pytest.approx(100.0)
