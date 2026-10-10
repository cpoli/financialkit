"""Tests for the square-root law, Bertsimas-Lo and Almgren-Chriss."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.market_microstructure import almgren_chriss, bertsimas_lo, execution_cost, fit_impact_law, square_root_impact


def test_square_root_law_and_fit():
    np.testing.assert_allclose(square_root_impact(4e4, 1e6, 0.02, Y=0.8), 0.8 * 0.02 * 0.2)
    rng = np.random.default_rng(0)
    q = 10 ** rng.uniform(-4, -1, 500)
    fit = fit_impact_law(q, square_root_impact(q, 1.0, 1.0, Y=0.6) * np.exp(0.3 * rng.standard_normal(500)))
    assert abs(fit.coef[1] - 0.5) < 4 * fit.std_errors[1]
    with pytest.raises(ValueError):
        fit_impact_law([0.0], [1.0])


def test_bertsimas_lo_without_information_is_twap():
    """Bertsimas-Lo (1998), Sec. 2.1: S_k = S / T and expected cost P0 S + theta S^2 (T + 1) / (2T)."""
    S, T, theta = 1e5, 10, 5e-5
    res = bertsimas_lo(S, T, theta, p0=50.0)
    np.testing.assert_allclose(res.trades, S / T, rtol=DEFAULT_RTOL)
    np.testing.assert_allclose(res.delta_w, 1 / (T - np.arange(T)))
    assert res.expected_cost == pytest.approx(50.0 * S + theta * S**2 * (T + 1) / (2 * T))


def test_bertsimas_lo_matches_brute_force_with_a_signal():
    """Without signal noise the problem is deterministic, so the DP policy is the constrained minimiser."""
    S, T, theta, gamma, rho, x0 = 1000.0, 5, 1e-3, 0.5, 0.7, 2.0
    res = bertsimas_lo(S, T, theta, gamma, rho, x0=x0)
    X = x0 * rho ** np.arange(1, T + 1)

    # Exact minimiser: cost = theta s'Ls + gamma cumsum(X)'s with L lower-triangular ones; solve the KKT system.
    L = np.tril(np.ones((T, T)))
    K = np.block([[theta * (L + L.T), -np.ones((T, 1))], [np.ones((1, T)), np.zeros((1, 1))]])
    s_star = np.linalg.solve(K, np.concatenate([-gamma * np.cumsum(X), [S]]))[:T]
    np.testing.assert_allclose(res.trades, s_star, rtol=1e-9)
    assert res.expected_cost == pytest.approx(float(np.sum((theta * np.cumsum(s_star) + gamma * np.cumsum(X)) * s_star)), rel=1e-10)
    assert res.trades[0] > S / T  # a rising price signal front-loads the purchase
    with pytest.raises(ValueError):
        bertsimas_lo(S, T, theta, signals=[1.0])


def test_almgren_chriss_is_efficient():
    """Almgren-Chriss (2000), Sec. 2: the sinh trajectory minimises E + lambda V among all schedules."""
    X, T, N, sigma, eta, gamma, eps, lam = 1e4, 5.0, 8, 0.95, 2.5e-6, 2.5e-7, 0.0625, 2e-6
    res = almgren_chriss(X, T, N, sigma, eta, lam, gamma, eps)

    # For a monotone schedule E + lambda V is quadratic in the inner holdings; its stationarity
    # condition (eta~/tau)(2x_k - x_k-1 - x_k+1) + lambda sigma^2 tau x_k = 0 is a tridiagonal system.
    tau = T / N
    c = (eta - 0.5 * gamma * tau) / tau
    A = np.diag(np.full(N - 1, 2 * c + lam * sigma**2 * tau)) - c * (np.eye(N - 1, k=1) + np.eye(N - 1, k=-1))
    rhs = np.zeros(N - 1)
    rhs[0] = c * X
    np.testing.assert_allclose(res.holdings[1:-1], np.linalg.solve(A, rhs), rtol=1e-9)
    assert (res.expected_cost, res.variance) == pytest.approx(execution_cost(res.holdings, T, sigma, eta, gamma, eps))


def test_almgren_chriss_limits():
    twap = almgren_chriss(100.0, 1.0, 4, 0.3, 0.1, 0.0)
    np.testing.assert_allclose(twap.trades, 25.0)
    urgent = almgren_chriss(100.0, 1.0, 4, 0.3, 0.1, 1e6)
    assert urgent.trades[0] > 99.0 and urgent.variance < twap.variance and urgent.expected_cost > twap.expected_cost
    with pytest.raises(ValueError):
        almgren_chriss(100.0, 1.0, 4, 0.3, 0.1, 1.0, gamma=1.0)
