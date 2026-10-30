"""Tests for Markowitz, Tobin and Merton mean-variance portfolios."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.portfolio import (
    efficient_frontier,
    efficient_portfolio,
    frontier_constants,
    min_variance_portfolio,
    tangency_portfolio,
    two_fund_allocation,
)

MU = np.array([0.05, 0.08, 0.12, 0.10])
VOL = np.array([0.10, 0.16, 0.25, 0.20])
CORR = np.array([[1.0, 0.3, 0.1, 0.2], [0.3, 1.0, 0.4, 0.5], [0.1, 0.4, 1.0, 0.3], [0.2, 0.5, 0.3, 1.0]])
COV = CORR * np.outer(VOL, VOL)
# SLSQP stops at a gradient tolerance, so constrained solutions agree to about 1e-6.
QP_TOL = 1e-6


def test_min_variance_closed_form_beats_random_portfolios():
    gmv = min_variance_portfolio(COV, MU)
    assert gmv.weights.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    for w in rng.normal(size=(200, 4)):
        w /= w.sum()
        assert np.sqrt(w @ COV @ w) >= gmv.volatility - DEFAULT_RTOL


def test_merton_constants_match_gmv_and_frontier():
    fc = frontier_constants(MU, COV)
    gmv = min_variance_portfolio(COV, MU)
    assert fc.gmv_return == pytest.approx(gmv.expected_return, rel=DEFAULT_RTOL)
    assert fc.gmv_volatility == pytest.approx(gmv.volatility, rel=DEFAULT_RTOL)
    for m in [0.06, 0.1, 0.2]:
        p = efficient_portfolio(MU, COV, m)
        assert p.expected_return == pytest.approx(m, rel=DEFAULT_RTOL)
        assert p.volatility == pytest.approx(fc.volatility(m), rel=DEFAULT_RTOL)
    assert fc.asymptote_slope() == pytest.approx(np.sqrt(fc.D / fc.C))
    with pytest.raises(ValueError):
        frontier_constants([0.1, 0.1], np.eye(2))


def test_unconstrained_efficient_portfolio_matches_slsqp_without_bounds():
    from financialkit.portfolio.utils import solve_qp

    w = solve_qp(COV, np.vstack([np.ones(4), MU]), [1.0, 0.1], bounds=(-5.0, 5.0))
    np.testing.assert_allclose(efficient_portfolio(MU, COV, 0.1).weights, w, atol=QP_TOL)


def test_long_only_frontier():
    fr = efficient_frontier(MU, COV, 12, long_only=True)
    assert np.all(fr.weights >= -QP_TOL)
    np.testing.assert_allclose(fr.weights.sum(axis=1), 1.0, atol=QP_TOL)
    assert fr.returns[-1] == pytest.approx(MU.max(), abs=QP_TOL)
    unconstrained = frontier_constants(MU, COV).volatility(fr.returns)
    assert np.all(fr.volatilities >= unconstrained - QP_TOL)
    with pytest.raises(ValueError):
        efficient_portfolio(MU, COV, 0.2, long_only=True)


def test_tangency_maximizes_sharpe_and_is_on_frontier():
    rf = 0.02
    t = tangency_portfolio(MU, COV, rf)
    fc = frontier_constants(MU, COV)
    assert t.volatility == pytest.approx(fc.volatility(t.expected_return), rel=DEFAULT_RTOL)
    # Merton (1972): the maximum Sharpe ratio is sqrt(B - 2 A rf + C rf^2).
    assert t.sharpe_ratio(rf) == pytest.approx(np.sqrt(fc.B - 2 * fc.A * rf + fc.C * rf**2), rel=DEFAULT_RTOL)


def test_long_only_tangency():
    mu = np.array([0.03, 0.08, 0.12])
    cov = np.array([[0.04, 0.0, 0.0], [0.0, 0.04, 0.035], [0.0, 0.035, 0.09]])
    t = tangency_portfolio(mu, cov, 0.02, long_only=True)
    assert np.all(t.weights >= -QP_TOL) and t.weights.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    for w in rng.dirichlet(np.ones(3), 300):
        assert (w @ mu - 0.02) / np.sqrt(w @ cov @ w) <= t.sharpe_ratio(0.02) + QP_TOL
    with pytest.raises(ValueError):
        tangency_portfolio(mu, cov, 0.2, long_only=True)


def test_two_fund_separation():
    """Tobin (1958): every investor holds the tangency mix; only the cash share varies."""
    rf = 0.02
    t = tangency_portfolio(MU, COV, rf).weights
    for gamma in [1.0, 3.0, 10.0]:
        res = two_fund_allocation(MU, COV, rf, gamma)
        np.testing.assert_allclose(res.weights / res.weights.sum(), t, rtol=DEFAULT_RTOL)
        assert res.expected_return == pytest.approx(rf + res.weights @ (MU - rf))
    with pytest.raises(ValueError):
        two_fund_allocation(MU, COV, rf, 0.0)


def test_solve_qp_respects_inequality_constraints_and_reports_infeasibility():
    from financialkit.portfolio.utils import solve_qp

    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    unconstrained = solve_qp(cov, np.ones((1, 2)), [1.0])
    assert unconstrained[1] == pytest.approx(4 / 13, abs=1e-8)  # inverse variance
    w = solve_qp(cov, np.ones((1, 2)), [1.0], A_ineq=[[0.0, 1.0]], b_ineq=[0.5])
    np.testing.assert_allclose(w, [0.5, 0.5], atol=1e-8)
    with pytest.raises(RuntimeError, match="quadratic program failed"):
        solve_qp(cov, np.ones((1, 2)), [1.0], bounds=(0.0, 0.1))
