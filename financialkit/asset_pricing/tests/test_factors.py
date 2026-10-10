"""Tests for factor regressions, Fama-MacBeth, the APT and characteristic sorts."""

import numpy as np
import pytest

from financialkit.asset_pricing import (
    apt_arbitrage_portfolio,
    apt_expected_returns,
    fama_macbeth,
    long_short_factor,
    momentum_signal,
    simulate_factor_model,
    sort_portfolios,
    time_series_regression,
)
from financialkit.constants import DEFAULT_ATOL


def test_time_series_regression_recovers_three_factor_loadings():
    B = np.array([[1.0, 0.5, -0.2], [0.8, -0.3, 0.6], [1.2, 0.0, 0.0]])
    V = np.diag([0.002, 0.001, 0.001])
    panel = simulate_factor_model(4000, B, [0.006, 0.002, 0.003], V, idio_vol=0.02, rng=0)
    res = time_series_regression(panel.excess_returns, panel.factors)
    assert res.betas.shape == (3, 3)
    np.testing.assert_allclose(res.betas, B, atol=0.03)
    assert np.all(np.abs(res.alpha_t) < 4)
    assert np.all((res.r_squared > 0) & (res.r_squared < 1))


def test_fama_macbeth_recovers_the_premium():
    betas = np.linspace(0.3, 1.8, 50)
    panel = simulate_factor_model(600, betas, [0.008], [[0.0025]], idio_vol=0.04, rng=1)
    res = fama_macbeth(panel.excess_returns, factors=panel.factors)
    # The cross-sectional slope estimates the realised factor mean, which is the premium for a traded factor.
    target = panel.factors.mean()
    assert abs(res.premia[1] - target) < 4 * res.std_errors[1]
    assert abs(res.premia[0]) < 4 * res.std_errors[0]
    np.testing.assert_allclose(res.t_values, res.premia / res.std_errors)


def test_fama_macbeth_with_given_betas_and_bad_arguments():
    betas = np.linspace(0.5, 1.5, 20)
    panel = simulate_factor_model(300, betas, [0.01], [[0.002]], rng=2)
    res = fama_macbeth(panel.excess_returns, betas=betas)
    assert res.gammas.shape == (300, 2)
    with pytest.raises(ValueError):
        fama_macbeth(panel.excess_returns)
    with pytest.raises(ValueError):
        fama_macbeth(panel.excess_returns, factors=panel.factors, betas=betas)


def test_apt_pricing_and_arbitrage():
    B = np.array([[1.0, 0.2], [0.5, 1.0], [1.5, -0.5], [0.8, 0.8]])
    lam = np.array([0.05, 0.02])
    mu = apt_expected_returns(B, lam, 0.03)
    np.testing.assert_allclose(apt_arbitrage_portfolio(mu, B), 0.0, atol=DEFAULT_ATOL)
    mu_bad = mu + np.array([0.0, 0.0, 0.0, 0.01])
    w = apt_arbitrage_portfolio(mu_bad, B)
    assert w.sum() == pytest.approx(0.0, abs=DEFAULT_ATOL)
    np.testing.assert_allclose(w @ B, 0.0, atol=DEFAULT_ATOL)
    assert w @ mu_bad == pytest.approx(w @ w) and w @ mu_bad > 0


def test_sorts_by_hand():
    R = np.array([[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.4]])
    S = np.array([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_allclose(sort_portfolios(S, R, 2), [[0.35, 0.15], [0.0, 0.2]])
    np.testing.assert_allclose(long_short_factor(S, R, 0.25), [-0.3, 0.4])
    sig = S.copy()
    sig[0] = np.nan
    assert np.isnan(sort_portfolios(sig, R, 2)[0]).all()
    assert np.isnan(long_short_factor(sig, R)[0])
    with pytest.raises(ValueError):
        long_short_factor(S, R, 0.6)


def test_momentum_signal_window():
    rng = np.random.default_rng(0)
    R = rng.normal(0.01, 0.05, (30, 3))
    sig = momentum_signal(R, lookback=12, skip=1)
    assert np.isnan(sig[:12]).all()
    np.testing.assert_allclose(sig[20], np.prod(1 + R[8:19], axis=0) - 1)
