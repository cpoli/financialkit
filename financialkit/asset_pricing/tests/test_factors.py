"""Tests for factor regressions, Fama-MacBeth, the APT and characteristic sorts."""

import numpy as np
import pytest

from financialkit.asset_pricing import (
    adjusted_p_values,
    apt_arbitrage_portfolio,
    apt_expected_returns,
    fama_french_five,
    fama_macbeth,
    long_short_factor,
    momentum_signal,
    simulate_factor_model,
    sort_portfolios,
    t_hurdle,
    time_series_regression,
    two_by_three_portfolios,
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


def test_two_by_three_portfolios_and_fama_french_five():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((24, 60))
    size, bm, op, inv = rng.standard_normal((4, 60))
    six = two_by_three_portfolios(size, bm, R)
    big = size > np.median(size)
    lo, hi = np.quantile(bm, [0.3, 0.7])
    np.testing.assert_allclose(six[:, 1, 2], R[:, big & (bm > hi)].mean(axis=1))
    np.testing.assert_allclose(six[:, 0, 0], R[:, ~big & (bm <= lo)].mean(axis=1))
    F = fama_french_five(R, R.mean(axis=1), size, bm, op, inv)
    assert F.shape == (24, 5)
    np.testing.assert_allclose(F[:, 2], six[:, :, 2].mean(axis=1) - six[:, :, 0].mean(axis=1))
    # CMA is long low investment: the reverse of a high-minus-low sort on investment.
    inv_six = two_by_three_portfolios(size, inv, R)
    np.testing.assert_allclose(F[:, 4], inv_six[:, :, 0].mean(axis=1) - inv_six[:, :, 2].mean(axis=1))


def test_adjusted_p_values():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    np.testing.assert_allclose(adjusted_p_values(p, "bonferroni"), np.minimum(10 * p, 1.0))
    holm = adjusted_p_values(p, "holm")
    assert holm[0] == pytest.approx(0.01) and holm[1] == pytest.approx(0.072)
    bh = adjusted_p_values(p, "bh")
    assert bh[0] == pytest.approx(0.01) and bh[1] == pytest.approx(0.04)
    c = np.sum(1 / np.arange(1, 11))
    np.testing.assert_allclose(adjusted_p_values(p, "bhy"), np.minimum(bh * c, 1.0))
    np.testing.assert_allclose(adjusted_p_values(p[::-1], "holm"), holm[::-1])
    with pytest.raises(ValueError, match="method"):
        adjusted_p_values(p, "sidak")


def test_t_hurdle():
    """Bonferroni with 316 tests: 3.78 (Harvey, Liu and Zhu 2016, Table 1, 2012)."""
    assert t_hurdle(np.zeros(316), method="bonferroni") == pytest.approx(3.78, abs=0.005)
    assert t_hurdle(np.zeros(10), method="holm") == np.inf
    t = np.array([5.0, 4.0, 0.5, -3.5, 0.1])
    assert t_hurdle(t, 0.05, "holm") == 3.5
