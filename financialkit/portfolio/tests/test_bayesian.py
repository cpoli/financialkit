"""Tests for Black-Litterman and Michaud's resampled frontier."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.portfolio import black_litterman, implied_equilibrium_returns, resampled_frontier

# He and Litterman (1999), Tables 1-2: Australia, Canada, France, Germany, Japan, UK, USA.
CORR = np.array(
    [
        [1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
        [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
        [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
        [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
        [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
        [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
        [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000],
    ]
)
VOL = np.array([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187])
COV = CORR * np.outer(VOL, VOL)
W_MKT = np.array([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615])
P_GERMANY = np.array([[0.0, 0.0, -0.295, 1.0, 0.0, -0.705, 0.0]])


def test_he_litterman_equilibrium_returns():
    """He and Litterman (1999), Table 2: implied equilibrium returns in percent."""
    pi = implied_equilibrium_returns(COV, W_MKT, 2.5)
    np.testing.assert_allclose(np.round(100 * pi, 1), [3.9, 6.9, 8.4, 9.0, 4.3, 6.8, 7.6])


def test_he_litterman_germany_view():
    """He and Litterman (1999), Table 4: Germany outperforms the rest of Europe by 5%."""
    res = black_litterman(COV, W_MKT, P_GERMANY, [0.05], tau=0.05, delta=2.5)
    np.testing.assert_allclose(np.round(100 * res.posterior_returns, 1), [4.3, 7.6, 9.3, 11.0, 4.5, 7.0, 8.1])
    np.testing.assert_allclose(np.round(100 * res.weights, 1), [1.5, 2.1, -3.9, 35.4, 11.0, -9.5, 58.6])


def test_weights_tilt_only_along_views():
    """He and Litterman: w* = w_mkt / (1 + tau) + P' Lambda, so the tilt lies in the span of P'."""
    P = np.vstack([P_GERMANY, np.eye(7)[1] - np.eye(7)[6]])
    res = black_litterman(COV, W_MKT, P, [0.05, 0.02])
    tilt = res.weights - W_MKT / (1 + res.tau)
    lam, *_ = np.linalg.lstsq(P.T, tilt, rcond=None)
    np.testing.assert_allclose(P.T @ lam, tilt, atol=1e-12)


def test_limits_of_view_confidence():
    vague = black_litterman(COV, W_MKT, P_GERMANY, [0.05], omega=[[1e6]])
    np.testing.assert_allclose(vague.posterior_returns, vague.prior_returns, atol=1e-8)
    np.testing.assert_allclose(vague.weights, W_MKT / 1.05, rtol=1e-6)
    sure = black_litterman(COV, W_MKT, P_GERMANY, [0.05], omega=[[1e-14]])
    assert (P_GERMANY @ sure.posterior_returns)[0] == pytest.approx(0.05, rel=DEFAULT_RTOL)
    with pytest.raises(ValueError):
        black_litterman(COV, W_MKT, P_GERMANY, [0.05, 0.01])


def test_resampled_frontier():
    mu, cov = np.array([0.06, 0.07, 0.08]), np.diag([0.02, 0.025, 0.03])
    res = resampled_frontier(mu, cov, n_obs=60, n_resamples=20, n_points=6, rng=0)
    assert res.weights.shape == (6, 3) and res.sample_weights.shape == (20, 6, 3)
    np.testing.assert_allclose(res.weights.sum(axis=1), 1.0, atol=1e-6)
    # Averaging spreads the top of the frontier across assets instead of a corner solution.
    assert res.weights[-1].max() < 0.99 and res.frontier.weights[-1].max() == pytest.approx(1.0, abs=1e-6)
    # Under the true inputs, averaged portfolios sit on or inside the true frontier.
    from financialkit.portfolio import efficient_portfolio

    for m, v in zip(res.returns, res.volatilities, strict=True):
        assert v >= efficient_portfolio(mu, cov, m, long_only=True).volatility - 1e-6
