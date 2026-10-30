"""Tests for Harrison-Kreps state prices and the Hansen-Jagannathan bound."""

import numpy as np
import pytest

from financialkit.asset_pricing import FiniteStateMarket, hansen_jagannathan_bound, max_sharpe_ratio
from financialkit.constants import DEFAULT_ATOL


def test_binomial_market_prices_a_call_like_the_one_period_tree():
    """Cox-Ross-Rubinstein one-step: q_u = (R - d) / (u - d) / R (Hull, 11th ed., Sec. 13.1)."""
    S0, u, d, R, K = 100.0, 1.2, 0.8, 1.05, 100.0
    mkt = FiniteStateMarket([[R, S0 * u], [R, S0 * d]], [1.0, S0])
    p = (R - d) / (u - d)
    np.testing.assert_allclose(mkt.risk_neutral_probabilities(), [p, 1 - p])
    assert mkt.risk_free_return() == pytest.approx(R)
    call = mkt.state_prices() @ np.maximum(S0 * np.array([u, d]) - K, 0)
    assert call == pytest.approx(p * 20 / R)
    lo, hi = mkt.price_bounds(np.maximum(S0 * np.array([u, d]) - K, 0))
    assert lo == pytest.approx(call) and hi == pytest.approx(call)


def test_sdf_prices_assets():
    probs = np.array([0.3, 0.7])
    mkt = FiniteStateMarket([[1.0, 2.0], [1.0, 0.5]], [0.95, 1.0], probs)
    m = mkt.sdf()
    np.testing.assert_allclose((probs * m) @ mkt.payoffs, mkt.prices)


def test_arbitrage_is_detected_and_exploited():
    X = np.array([[1.0, 2.0], [1.0, 0.5]])
    mkt = FiniteStateMarket(X, [0.95, 0.4])  # stock too cheap: it pays at least 0.5 > 0.4 / 0.95 bonds' worth
    assert not mkt.is_arbitrage_free()
    theta = mkt.arbitrage_portfolio()
    assert np.all(X @ theta >= -DEFAULT_ATOL)
    assert mkt.prices @ theta <= DEFAULT_ATOL
    assert (X @ theta).sum() - mkt.prices @ theta > 0
    with pytest.raises(ValueError):
        mkt.state_prices()
    assert np.all(FiniteStateMarket(X, [0.95, 1.0]).arbitrage_portfolio() == 0)


def test_incomplete_trinomial_market_gives_a_price_interval():
    S = np.array([120.0, 100.0, 80.0])
    mkt = FiniteStateMarket(np.column_stack([np.ones(3), S]), [1 / 1.02, 100.0])
    assert not mkt.is_complete and mkt.is_arbitrage_free()
    q = mkt.state_prices()
    assert np.all(q > 0)
    np.testing.assert_allclose(q @ mkt.payoffs, mkt.prices)
    lo, hi = mkt.price_bounds(np.maximum(S - 100, 0))
    assert lo < q @ np.maximum(S - 100, 0) < hi
    # Merton's bound: a call is worth at least S - K / R; the upper bound comes from the extreme state prices.
    assert lo == pytest.approx(100 - 100 / 1.02)


def test_hj_bound_equals_sharpe_ratio_identity():
    """sigma_min(v) / v is the largest Sharpe ratio when the zero-beta rate is 1 / v (Hansen-Jagannathan 1991, eq. 15)."""
    mu = np.array([1.06, 1.09, 1.12])
    vol = np.array([0.12, 0.18, 0.25])
    C = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.5], [0.2, 0.5, 1.0]]) * np.outer(vol, vol)
    v = np.linspace(0.9, 1.0, 11)
    b = hansen_jagannathan_bound(mu, C, v)
    sr = np.array([max_sharpe_ratio(mu - 1 / vi, C) for vi in v])
    np.testing.assert_allclose(b.sdf_stds / v, sr)


def test_hj_minimum_variance_sdf_prices_the_returns():
    rng = np.random.default_rng(0)
    mu = np.array([1.05, 1.08])
    C = np.array([[0.02, 0.006], [0.006, 0.05]])
    R = rng.multivariate_normal(mu, C, 400_000)
    v = 0.97
    m = v + (R - mu) @ np.linalg.solve(C, 1 - v * mu)
    x = m[:, None] * R
    se = x.std(axis=0) / np.sqrt(len(m))
    assert np.all(np.abs(x.mean(axis=0) - 1) < 4 * se)
    b = hansen_jagannathan_bound(mu, C, [v])
    assert m.std() == pytest.approx(b.sdf_stds[0], rel=0.01)
    assert b.admits(v, b.sdf_stds[0] * 1.01) and not b.admits(v, b.sdf_stds[0] * 0.99)


def test_law_of_one_price_violation_has_no_state_prices():
    """Two identical payoffs at different prices: X'q = p has no solution, so the market is not arbitrage-free."""
    market = FiniteStateMarket([[1.0, 1.0], [1.0, 1.0]], [0.90, 0.95])
    assert not market.is_arbitrage_free()
    with pytest.raises(ValueError, match="one entry per payoff column"):
        FiniteStateMarket([[1.0, 2.0], [1.0, 0.5]], [0.95])
