"""Tests for Kyle's equilibrium and the Glosten-Milgrom dealer."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_ATOL
from financialkit.market_microstructure import glosten_milgrom_quotes, kyle_equilibrium, kyle_lambda, simulate_glosten_milgrom, simulate_kyle


def test_kyle_equilibrium_closed_form():
    """Kyle (1985), Theorem 1: lambda = sigma_v / (2 sigma_u), beta = sigma_u / sigma_v."""
    eq = kyle_equilibrium(3.0, 2.0, p0=10.0)
    assert (eq.lam, eq.beta, eq.insider_profit, eq.posterior_variance) == (0.75, 2 / 3, 3.0, 4.5)
    assert eq.beta == pytest.approx(1 / (2 * eq.lam))  # the insider's best response to lambda
    with pytest.raises(ValueError):
        kyle_equilibrium(0.0, 1.0)


def test_kyle_simulation_is_consistent():
    """The market maker's lambda is the regression slope of v on order flow, and the insider earns sigma_v sigma_u / 2."""
    eq = kyle_equilibrium(1.0, 2.0, p0=5.0)
    v, y, p = simulate_kyle(eq, 20000, rng=0)
    fit = kyle_lambda(v - eq.p0, y)
    assert abs(fit.coef[1] - eq.lam) < 4 * fit.std_errors[1]
    profit = eq.beta * (v - eq.p0) * (v - p)
    assert abs(profit.mean() - eq.insider_profit) < 4 * profit.std() / np.sqrt(v.size)
    resid = v - p
    assert abs(resid.var() - eq.posterior_variance) < 4 * eq.posterior_variance * np.sqrt(2 / v.size)


def test_kyle_insider_cannot_do_better():
    """Given lambda, expected profit beta sigma_v^2 (1 - lambda beta) peaks at the equilibrium beta."""
    eq = kyle_equilibrium(1.0, 1.0)
    betas = np.linspace(0.2, 2.0, 19)
    profits = betas * eq.sigma_v**2 * (1 - eq.lam * betas)
    assert betas[np.argmax(profits)] == pytest.approx(eq.beta)


def test_glosten_milgrom_quotes_are_conditional_expectations():
    """Glosten-Milgrom (1985), Sec. 2: the belief is a martingale and the spread grows with the informed share."""
    theta, alpha, eta = 0.3, 0.25, 0.4
    b, a = glosten_milgrom_quotes(theta, alpha, 0.0, 1.0, eta)
    p_buy = theta * (alpha + (1 - alpha) * eta) + (1 - theta) * (1 - alpha) * eta
    assert p_buy * a + (1 - p_buy) * b == pytest.approx(theta)
    assert b < theta < a
    b0, a0 = glosten_milgrom_quotes(theta, 0.0, 0.0, 1.0, eta)
    assert a0 - b0 == pytest.approx(0.0, abs=DEFAULT_ATOL)
    spreads = [np.subtract(*glosten_milgrom_quotes(0.5, a, 0.0, 1.0)[::-1]) for a in (0.1, 0.3, 0.6)]
    assert np.all(np.diff(spreads) > 0)
    with pytest.raises(ValueError):
        glosten_milgrom_quotes(0.5, 1.0, 0.0, 1.0)


def test_glosten_milgrom_prices_converge_to_value():
    res = simulate_glosten_milgrom(300, 0.3, 90.0, 110.0, true_value=110.0, rng=0)
    assert res.belief[-1] > 0.999 and res.ask[-1] - res.bid[-1] < 0.05
    assert np.all(res.bid <= res.ask)
    with pytest.raises(ValueError):
        simulate_glosten_milgrom(5, 0.3, 90.0, 110.0, true_value=100.0)
