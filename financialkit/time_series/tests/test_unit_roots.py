"""Tests for MacKinnon critical values, the ADF test and the Engle-Granger test."""

import numpy as np
import pytest

from financialkit.time_series import adf_test, engle_granger, johansen_test, kpss_test, mackinnon_critical_values, simulate_arma
from financialkit.time_series.utils import simulate_cointegrated_pair, simulate_dickey_fuller


def test_asymptotic_critical_values_match_published_tables():
    """Fuller (1976, Table 8.5.2) asymptotic 5% values: -1.95, -2.86, -3.41; Engle and Yoo (1987)
    two-variable cointegration 5% value: -3.34."""
    inf = 10**12
    assert mackinnon_critical_values(inf, "n")["5%"] == pytest.approx(-1.94, abs=0.01)
    assert mackinnon_critical_values(inf, "c")["5%"] == pytest.approx(-2.86, abs=0.01)
    assert mackinnon_critical_values(inf, "ct")["5%"] == pytest.approx(-3.41, abs=0.01)
    assert mackinnon_critical_values(inf, "c", n_vars=2)["5%"] == pytest.approx(-3.34, abs=0.01)
    cv = mackinnon_critical_values(100, "c")
    assert cv["1%"] < cv["5%"] < cv["10%"]
    with pytest.raises(ValueError):
        mackinnon_critical_values(100, "n", n_vars=2)


@pytest.mark.parametrize("regression", ["n", "c", "ct"])
def test_dickey_fuller_size_at_mackinnon_critical_values(regression):
    """Simulated Dickey-Fuller statistics fall below the 5% and 1% critical values with
    the nominal frequency (binomial standard error)."""
    n_sims, T = 20_000, 150
    stats = simulate_dickey_fuller(T, n_sims, regression, rng=0)
    cv = mackinnon_critical_values(T, regression)
    for level, p in (("5%", 0.05), ("1%", 0.01)):
        assert abs(np.mean(stats < cv[level]) - p) < 4 * np.sqrt(p * (1 - p) / n_sims)


@pytest.mark.parametrize("n_vars", [2, 3])
def test_engle_granger_size_on_independent_random_walks(n_vars):
    rng = np.random.default_rng(n_vars)
    n_sims, T = 3000, 200
    hits = []
    for _ in range(n_sims):
        walks = np.cumsum(rng.standard_normal((T, n_vars)), axis=0)
        hits.append(engle_granger(walks[:, 0], walks[:, 1:], lags=0).rejects("5%"))
    assert abs(np.mean(hits) - 0.05) < 4 * np.sqrt(0.05 * 0.95 / n_sims)


def test_adf_without_lags_equals_simulated_statistic():
    for regression in ("n", "c", "ct"):
        y = np.cumsum(np.random.default_rng(1).standard_normal(101))
        stat = simulate_dickey_fuller(100, 1, regression, rng=1)[0]
        assert adf_test(y, lags=0, regression=regression).statistic == pytest.approx(stat, rel=1e-10)


def test_adf_detects_stationarity_and_selects_lags():
    x = simulate_arma(ar=[0.5, 0.3], n=1000, rng=2)
    res = adf_test(x)
    assert res.rejects("1%")
    assert 1 <= res.lags <= int(12 * (1000 / 100) ** 0.25)
    se = res.gamma / res.statistic  # the statistic is the t-ratio of gamma
    assert abs(res.gamma - (0.5 + 0.3 - 1)) < 4 * se
    walk = np.cumsum(simulate_arma(ar=[0.5], n=1000, rng=3))
    assert not adf_test(walk).rejects("10%")
    with pytest.raises(ValueError):
        adf_test(walk, lags=0, regression="x")


def test_engle_granger_finds_cointegration_and_rejects_spurious_regression():
    y, x = simulate_cointegrated_pair(2000, beta=1.5, intercept=3.0, phi=0.9, rng=4)
    res = engle_granger(y, x)
    assert res.rejects("1%")
    assert res.beta[0] == pytest.approx(1.5, abs=0.02)
    assert res.intercept == pytest.approx(3.0, abs=1.0)
    rng = np.random.default_rng(5)
    a, b = np.cumsum(rng.standard_normal(2000)), np.cumsum(rng.standard_normal(2000))
    spurious = engle_granger(a, b)
    assert not spurious.rejects("5%")
    assert spurious.critical_values == mackinnon_critical_values(spurious.adf.n_obs, "c", n_vars=2)


def test_kpss_statistic_matches_the_formula_and_table():
    """With no lags the long-run variance is the residual variance; critical values from KPSS (1992), Table 1."""
    y = np.random.default_rng(6).standard_normal(200)
    res = kpss_test(y, lags=0)
    e = y - y.mean()
    assert res.statistic == pytest.approx(np.sum(np.cumsum(e) ** 2) / (200**2 * np.mean(e**2)), rel=1e-12)
    assert res.critical_values == {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}
    assert kpss_test(y, "ct").critical_values["5%"] == 0.146
    assert res.lags == 0 and kpss_test(y).lags == int(12 * 2**0.25)
    with pytest.raises(ValueError):
        kpss_test(y, "n")


def test_kpss_pvalue_interpolates_and_clips():
    res = kpss_test(np.random.default_rng(7).standard_normal(100))
    for stat, p in ((0.463, 0.05), (0.01, 0.10), (5.0, 0.01)):
        res.statistic = stat
        assert res.p_value == pytest.approx(p)


@pytest.mark.parametrize("regression", ["c", "ct"])
def test_kpss_size_under_stationarity(regression):
    rng = np.random.default_rng(8)
    n_sims = 2000
    trend = np.arange(300.0) * 0.01 if regression == "ct" else 0.0
    hits = [kpss_test(trend + rng.standard_normal(300), regression, lags=4).rejects("5%") for _ in range(n_sims)]
    assert abs(np.mean(hits) - 0.05) < 4 * np.sqrt(0.05 * 0.95 / n_sims)


def test_kpss_rejects_random_walks_and_needs_the_trend_term():
    rng = np.random.default_rng(9)
    assert kpss_test(np.cumsum(rng.standard_normal(1000))).rejects("1%")
    trend_stationary = 0.05 * np.arange(1000) + rng.standard_normal(1000)
    assert kpss_test(trend_stationary, "c").rejects("1%")
    assert not kpss_test(trend_stationary, "ct").rejects("5%")


@pytest.mark.parametrize(("regression", "drift"), [("n", 0.0), ("c", 0.2)])
def test_johansen_size_on_independent_random_walks(regression, drift):
    """Under rank 0, the trace and maximum-eigenvalue tests reject at about their nominal 5%
    (MacKinnon-Haug-Michelis 1999 asymptotic values; T = 500)."""
    rng = np.random.default_rng(10)
    n_sims = 2000
    trace, maxev = [], []
    for _ in range(n_sims):
        res = johansen_test(np.cumsum(drift + rng.standard_normal((500, 2)), axis=0), lags=0, regression=regression)
        trace.append(res.trace[0] > res.trace_critical_values["5%"][0])
        maxev.append(res.max_eigen[0] > res.max_eigen_critical_values["5%"][0])
    se = np.sqrt(0.05 * 0.95 / n_sims)
    assert abs(np.mean(trace) - 0.05) < 4 * se
    assert abs(np.mean(maxev) - 0.05) < 4 * se


def test_johansen_finds_the_rank_and_the_cointegrating_vector():
    rng = np.random.default_rng(11)
    T = 2000
    w = np.cumsum(0.1 + rng.standard_normal((T, 2)), axis=0)
    u = simulate_arma(ar=[0.5], n=T, rng=12)
    y = np.column_stack([w[:, 0], w[:, 1], 0.5 * w[:, 0] - 1.5 * w[:, 1] + u])
    res = johansen_test(y, lags=1)
    assert res.rank("5%") == 1 and res.rank("5%", test="max_eigen") == 1
    v = res.cointegrating_vectors(1)[:, 0]
    np.testing.assert_allclose(v, [1.0, -3.0, -2.0], atol=0.05)  # (0.5, -1.5, -1) scaled to a leading 1
    np.testing.assert_allclose(res.trace, np.cumsum(res.max_eigen[::-1])[::-1])
    assert np.all(np.diff(res.eigenvalues) <= 0) and res.n_obs == T - 2
    # Critical values grow with the number of nonstationary directions.
    assert np.all(np.diff(res.trace_critical_values["5%"]) < 0)
    # Two independent stationary series have full rank.
    assert johansen_test(rng.standard_normal((500, 2)), lags=0).rank() == 2


def test_johansen_input_checks():
    with pytest.raises(ValueError):
        johansen_test(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        johansen_test(np.zeros((100, 6)))
    with pytest.raises(ValueError):
        johansen_test(np.random.default_rng(0).standard_normal((100, 2)), regression="ct")
    with pytest.raises(ValueError):
        johansen_test(np.random.default_rng(0).standard_normal((8, 2)), lags=3)
