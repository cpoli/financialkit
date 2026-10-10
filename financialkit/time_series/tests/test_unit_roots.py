"""Tests for MacKinnon critical values, the ADF test and the Engle-Granger test."""

import numpy as np
import pytest

from financialkit.time_series import adf_test, engle_granger, mackinnon_critical_values, simulate_arma
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
    assert spurious.critical_values == mackinnon_critical_values(spurious.adf.nobs, "c", n_vars=2)
