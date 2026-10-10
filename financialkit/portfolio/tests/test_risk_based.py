"""Tests for CVaR optimization, maximum diversification, risk parity and HRP."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.portfolio import (
    cvar_optimization,
    diversification_ratio,
    hierarchical_risk_parity,
    inverse_volatility_portfolio,
    min_variance_portfolio,
    most_diversified_portfolio,
    risk_parity,
)
from financialkit.risk import euler_allocation, expected_shortfall, historical_var

VOL = np.array([0.10, 0.15, 0.20, 0.30])
CORR = np.array([[1.0, 0.2, 0.4, 0.1], [0.2, 1.0, 0.3, 0.5], [0.4, 0.3, 1.0, 0.6], [0.1, 0.5, 0.6, 1.0]])
COV = CORR * np.outer(VOL, VOL)


def test_cvar_lp_matches_scenario_es_and_var():
    rng = np.random.default_rng(0)
    R = rng.multivariate_normal([0.01, 0.02, 0.015, 0.03], COV / 12, size=1000)
    res = cvar_optimization(R, 0.95)
    pnl = R @ res.weights
    assert res.cvar == pytest.approx(expected_shortfall(pnl, 0.95), rel=1e-7)
    # Rockafellar-Uryasev, Theorem 1: the optimal zeta is (the left end of the interval of) the VaR.
    assert res.var == pytest.approx(historical_var(pnl, 0.95).var, abs=np.diff(np.sort(pnl)).max())
    for w in rng.dirichlet(np.ones(4), 100):
        assert expected_shortfall(R @ w, 0.95) >= res.cvar - 1e-10
    constrained = cvar_optimization(R, 0.95, target_return=0.025)
    assert constrained.expected_return >= 0.025 - 1e-9 and constrained.cvar >= res.cvar
    unconstrained = cvar_optimization(R, 0.95, long_only=False)
    assert unconstrained.cvar <= res.cvar + 1e-12


def test_most_diversified_portfolio():
    unc = most_diversified_portfolio(COV, long_only=False)
    y = np.linalg.solve(COV, VOL)
    np.testing.assert_allclose(unc.weights, y / y.sum(), rtol=DEFAULT_RTOL)
    lo = most_diversified_portfolio(COV)
    dr = diversification_ratio(lo.weights, COV)
    rng = np.random.default_rng(1)
    for w in rng.dirichlet(np.ones(4), 300):
        assert diversification_ratio(w, COV) <= dr + 1e-6
    # Choueifaty-Coignard (2008), core property: every held asset has the same correlation with the MDP.
    held = lo.weights > 1e-6
    corr_with_mdp = (COV @ lo.weights) / (VOL * lo.volatility)
    assert np.ptp(corr_with_mdp[held]) < 1e-5
    assert np.all(corr_with_mdp[~held] >= corr_with_mdp[held].min() - 1e-6)


def test_risk_parity_equalizes_euler_contributions():
    rp = risk_parity(COV)
    shares = euler_allocation(rp.weights, COV, measure="volatility").shares
    np.testing.assert_allclose(shares, 0.25, rtol=1e-8)
    budgets = np.array([0.4, 0.3, 0.2, 0.1])
    np.testing.assert_allclose(euler_allocation(risk_parity(COV, budgets).weights, COV, measure="volatility").shares, budgets, rtol=1e-8)
    # Maillard-Roncalli-Teiletche (2010): sigma_MV <= sigma_ERC <= sigma_1/N.
    ew = np.full(4, 0.25)
    assert min_variance_portfolio(COV).volatility <= rp.volatility <= np.sqrt(ew @ COV @ ew)
    # With a constant correlation, ERC is inverse volatility (MRT 2010, Sec. 2).
    cc = (0.3 + 0.7 * np.eye(4)) * np.outer(VOL, VOL)
    np.testing.assert_allclose(risk_parity(cc).weights, inverse_volatility_portfolio(cc).weights, rtol=1e-8)
    with pytest.raises(ValueError):
        risk_parity(COV, [1.0, 0.0, 1.0, 1.0])


def test_hrp():
    """Block structure: two clusters split by inverse cluster variance."""
    block = np.kron(np.eye(2), 0.8 * np.ones((3, 3))) + 0.2 * np.eye(6)
    vol = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    cov = block * np.outer(vol, vol)
    res = hierarchical_risk_parity(cov)
    assert res.weights.sum() == pytest.approx(1.0)
    assert set(res.order[:3]) in ({0, 1, 2}, {3, 4, 5})
    v1 = np.full(3, 1 / 3) @ cov[:3, :3] @ np.full(3, 1 / 3)
    v2 = np.full(3, 1 / 3) @ cov[3:, 3:] @ np.full(3, 1 / 3)
    assert res.weights[:3].sum() == pytest.approx(v2 / (v1 + v2))
    diag = np.diag([0.01, 0.04, 0.09, 0.02, 0.05])
    ivp = 1 / np.diag(diag)
    np.testing.assert_allclose(hierarchical_risk_parity(diag).weights, ivp / ivp.sum(), rtol=DEFAULT_RTOL)
