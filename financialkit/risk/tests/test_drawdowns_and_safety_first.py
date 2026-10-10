"""Tests for drawdowns, the Calmar ratio and Roy's safety-first criterion."""

import numpy as np
import pytest
from scipy import stats

from financialkit.constants import DEFAULT_RTOL
from financialkit.risk import calmar_ratio, drawdowns, safety_first_portfolio, safety_first_ratio, shortfall_probability


def test_drawdown_path_by_hand():
    res = drawdowns([0.1, -0.2, 0.05, 0.25, -0.1])
    np.testing.assert_allclose(res.wealth, [1.0, 1.1, 0.88, 0.924, 1.155, 1.0395])
    assert res.max_drawdown == pytest.approx(0.2)
    assert (res.peak_index, res.trough_index, res.recovery_index) == (1, 2, 4)
    assert res.longest_duration == 2


def test_drawdowns_of_log_returns_and_monotone_series():
    r = np.log([1.1, 0.9])
    assert drawdowns(r, log=True).max_drawdown == pytest.approx(0.1)
    up = drawdowns([0.01] * 5)
    assert up.max_drawdown == 0.0 and up.recovery_index == -1
    assert calmar_ratio([0.01] * 12, periods_per_year=12) == np.inf


def test_calmar_ratio():
    r = [0.10, -0.05, 0.08, 0.02]
    assert calmar_ratio(r, periods_per_year=4) == pytest.approx((1.1 * 0.95 * 1.08 * 1.02 - 1) / 0.05)


def test_expected_max_drawdown_of_brownian_motion():
    """Magdon-Ismail et al. (2004): for zero drift, E[MDD of log wealth] = sqrt(pi / 2) sigma sqrt(T)."""
    rng = np.random.default_rng(0)
    m, n, sigma = 2000, 2000, 0.2
    r = sigma * np.sqrt(1 / n) * rng.standard_normal((m, n))
    mdd = np.array([-np.log(1 - drawdowns(row, log=True).max_drawdown) for row in r])
    se = mdd.std() / np.sqrt(m)
    # Discrete monitoring shortens the drawdown by about 0.5826 sigma sqrt(dt) (Broadie-Glasserman-Kou).
    exact = np.sqrt(np.pi / 2) * sigma - 0.5826 * sigma * np.sqrt(1 / n)
    assert abs(mdd.mean() - exact) < 4 * se + 0.5826 * sigma * np.sqrt(1 / n)


def test_safety_first_ratio_and_bounds():
    assert safety_first_ratio(0.08, 0.15, 0.02) == pytest.approx(0.4)
    s = np.array([0.5, 2.0, 4.0])
    assert np.all(shortfall_probability(s, 1.0, 0.0) >= shortfall_probability(s, 1.0, 0.0, "normal"))
    assert shortfall_probability(-0.1, 0.2, 0.0) == 1.0
    assert shortfall_probability(0.1, 0.2, 0.0, "normal") == pytest.approx(stats.norm.cdf(-0.5))
    with pytest.raises(ValueError):
        shortfall_probability(0.1, 0.2, bound="nope")


def test_safety_first_portfolio_maximises_roy_ratio():
    mu = np.array([0.05, 0.08, 0.12])
    cov = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.02], [0.0, 0.02, 0.16]])
    d = 0.01
    w = safety_first_portfolio(mu, cov, d)
    assert w.sum() == pytest.approx(1.0)
    best = safety_first_ratio(w @ mu, np.sqrt(w @ cov @ w), d)
    rng = np.random.default_rng(1)
    for v in rng.dirichlet(np.ones(3), 200):
        assert safety_first_ratio(v @ mu, np.sqrt(v @ cov @ v), d) <= best + DEFAULT_RTOL
    with pytest.raises(ValueError):
        safety_first_portfolio(mu, cov, 0.2)
