"""Tests for Garman's dealer, Ho-Stoll quotes and Avellaneda-Stoikov market making."""

import numpy as np
import pytest
from scipy import optimize

from financialkit.constants import DEFAULT_ATOL, DEFAULT_RTOL
from financialkit.market_microstructure import (
    AvellanedaStoikov,
    SymmetricQuotes,
    avellaneda_stoikov_spread,
    garman_ruin_probability,
    ho_stoll_quotes,
    reservation_price,
    simulate_garman,
    simulate_market_making,
)


def test_garman_ruin_probability_matches_simulation():
    """Gambler's ruin (Garman 1976, Sec. 3): P(ruin | I0) = (lambda_a / lambda_b)^I0."""
    res = simulate_garman(3, np.inf, 1.0, 1.0, 2.0, 3.0, n_events=1000, n_paths=4000, rng=0)
    p = res.failed.mean()
    exact = float(garman_ruin_probability(3, 2.0, 3.0))
    assert abs(p - exact) < 4 * np.sqrt(exact * (1 - exact) / res.failed.size)
    assert garman_ruin_probability(5, 3.0, 2.0) == 1.0
    with pytest.raises(ValueError):
        garman_ruin_probability(-1, 1.0, 1.0)


def test_garman_needs_a_spread():
    """With equal quotes, stock and cash cannot both drift up, so most dealers fail; a spread saves them."""
    # At a = b = 1 cash + stock is constant, so the inventory is a driftless walk trapped between two ruins.
    no_spread = simulate_garman(10, 10.0, 1.0, 1.0, 1.0, 1.0, n_events=5000, n_paths=400, rng=1)
    spread = simulate_garman(10, 10.0, 1.2, 0.8, 1.0, 1.25, n_events=5000, n_paths=400, rng=1)
    assert no_spread.failed.all()
    assert spread.failed.mean() < 0.5
    # Paths are frozen after failure.
    j = int(np.argmax(no_spread.failed))
    k = int(np.searchsorted(no_spread.times[j], no_spread.failure_time[j]))
    assert np.all(no_spread.stock[j, k:] == no_spread.stock[j, k])


def test_ho_stoll_quotes_solve_the_dealer_problem():
    """The closed form maximises lambda_a d_a + lambda_b d_b - (gamma/2) s^2 tau (I + lambda_b - lambda_a)^2."""
    alpha, beta, g, sigma, tau, I = 10.0, 20.0, 0.5, 0.3, 2.0, 7.0

    def neg(d):
        la, lb = alpha - beta * d[0], alpha - beta * d[1]
        return -(la * d[0] + lb * d[1] - 0.5 * g * sigma**2 * tau * (I + lb - la) ** 2)

    da, db = optimize.minimize(neg, [0.2, 0.2], method="BFGS", options={"gtol": 1e-10}).x
    b, a = ho_stoll_quotes(0.0, I, alpha, beta, g, sigma, tau)
    np.testing.assert_allclose([a, -b], [da, db], atol=1e-6)
    bb, aa = ho_stoll_quotes(0.0, [-5.0, 0.0, 5.0], alpha, beta, g, sigma, tau)
    np.testing.assert_allclose(aa - bb, alpha / beta, rtol=DEFAULT_RTOL)
    assert np.all(np.diff(aa) < 0)


def test_avellaneda_stoikov_formulas():
    """Avellaneda-Stoikov (2008), eqs. (29)-(30)."""
    assert reservation_price(100.0, 0.0, 0.1, 2.0, 1.0) == 100.0
    np.testing.assert_allclose(reservation_price(100.0, 3.0, 0.1, 2.0, 0.5), 100.0 - 3 * 0.1 * 4 * 0.5)
    np.testing.assert_allclose(avellaneda_stoikov_spread(1e-8, 2.0, 1.0, 1.5), 2 / 1.5, rtol=1e-6)
    b, a = AvellanedaStoikov(0.1, 2.0, 1.5, 1.0).quotes(np.array([100.0]), np.array([0.0]), 1.0)
    np.testing.assert_allclose(a - b, 20 * np.log1p(0.1 / 1.5), atol=DEFAULT_ATOL)
    with pytest.raises(ValueError):
        avellaneda_stoikov_spread(0.0, 2.0, 1.0, 1.5)


def test_inventory_strategy_has_lower_risk_than_symmetric():
    """Avellaneda-Stoikov (2008), Table 1 (gamma = 0.1): the inventory strategy roughly halves the profit
    standard deviation and cuts the terminal inventory spread, against a symmetric strategy with the same average spread."""
    T, n, paths = 1.0, 200, 1000
    inv = simulate_market_making(AvellanedaStoikov(0.1, 2.0, 1.5, T), 100.0, 2.0, T, n, 140.0, 1.5, paths, rng=0)
    half = float(np.mean(inv.ask - inv.bid)) / 2
    sym = simulate_market_making(SymmetricQuotes(half), 100.0, 2.0, T, n, 140.0, 1.5, paths, rng=0)
    assert inv.pnl.std() < 0.6 * sym.pnl.std()
    assert inv.inventory[:, -1].std() < 0.5 * sym.inventory[:, -1].std()
    # Cash and inventory always reconcile with the fills.
    np.testing.assert_allclose(sym.ask - sym.bid, 2 * half)
    assert inv.mid.shape == (paths, n + 1) and inv.bid.shape == (paths, n)
