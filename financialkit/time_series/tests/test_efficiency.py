"""Tests for autocorrelations, the runs test, portmanteau statistics and variance ratios."""

import numpy as np
import pytest
from scipy.signal import lfilter

from financialkit.time_series import (
    GARCH,
    autocorrelation,
    autocovariance,
    bartlett_band,
    box_pierce,
    ljung_box,
    runs_test,
    variance_ratio,
)


def _rejection_rate_ok(rejections, n, level=0.05):
    return abs(np.mean(rejections) - level) < 4 * np.sqrt(level * (1 - level) / n)


def test_autocovariance_matches_direct_sum():
    x = np.random.default_rng(0).standard_normal(257)
    d = x - x.mean()
    direct = np.array([d[k:] @ d[: d.size - k] / d.size for k in range(6)])
    np.testing.assert_allclose(autocovariance(x, 5), direct, rtol=1e-12, atol=1e-14)
    assert autocorrelation(x, 5)[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        autocovariance(x, 257)


def test_random_walk_levels_versus_changes():
    """Kendall's contrast: price levels are highly autocorrelated, their changes are not."""
    changes = np.random.default_rng(1).standard_normal(5000)
    assert autocorrelation(np.cumsum(changes), 5)[1:].min() > 0.9
    assert np.all(np.abs(autocorrelation(changes, 10)[1:]) < 4 / np.sqrt(changes.size))
    assert bartlett_band(10_000, 0.95) == pytest.approx(1.959964 / 100, rel=1e-6)


def test_runs_test_two_signs_reduces_to_wald_wolfowitz():
    """With no zero changes, Fama's mean and variance equal the Wald-Wolfowitz formulas."""
    x = np.random.default_rng(2).standard_normal(101)
    n1, n2 = np.sum(x > 0), np.sum(x < 0)
    N = n1 + n2
    m = 2 * n1 * n2 / N + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - N) / (N**2 * (N - 1))
    runs = 1 + np.count_nonzero(np.sign(x[1:]) != np.sign(x[:-1]))
    diff = runs - m
    diff -= np.sign(diff) * min(0.5, abs(diff))
    assert runs_test(x).statistic == pytest.approx(diff / np.sqrt(var), rel=1e-12)


def test_runs_test_size_and_power():
    rng = np.random.default_rng(3)
    reject = [runs_test(rng.standard_normal(200)).rejects() for _ in range(400)]
    assert _rejection_rate_ok(reject, 400)
    trending = lfilter([1.0], [1.0, -0.5], rng.standard_normal(500))
    res = runs_test(trending)
    assert res.statistic < 0 and res.rejects(0.001)
    assert runs_test(np.r_[np.zeros(5), np.ones(5), -np.ones(5)]).n_obs == 15
    with pytest.raises(ValueError):
        runs_test(np.ones(10))


def test_ljung_box_matches_formula_and_exceeds_box_pierce():
    x = lfilter([1.0], [1.0, -0.2], np.random.default_rng(4).standard_normal(300))
    rho = autocorrelation(x, 6)[1:]
    n, k = x.size, np.arange(1, 7)
    lb = ljung_box(x, 6, fitted_params=1)
    assert lb.statistic == pytest.approx(n * (n + 2) * np.sum(rho**2 / (n - k)), rel=1e-12)
    assert lb.df == 5
    assert box_pierce(x, 6).statistic < lb.statistic
    with pytest.raises(ValueError):
        ljung_box(x, 2, fitted_params=2)


def test_ljung_box_size_under_white_noise():
    rng = np.random.default_rng(5)
    reject = [ljung_box(rng.standard_normal(500), 10).rejects() for _ in range(400)]
    assert _rejection_rate_ok(reject, 400)


def test_variance_ratio_of_ar1_returns_matches_population_value():
    """VR(q) = 1 + 2 sum_{k<q} (1 - k/q) rho_k for returns with autocorrelations rho_k = phi^k.
    The tolerance uses the random-walk standard error, which is close to the true one for
    phi = 0.1 (Lo and MacKinlay 1988, Sec. 2)."""
    phi, T = 0.1, 200_000
    x = lfilter([1.0], [1.0, -phi], np.random.default_rng(6).standard_normal(T))
    res = variance_ratio(np.r_[0.0, np.cumsum(x)], q=[2, 4, 8])
    for q, vr in zip(res.q, res.ratio, strict=True):
        k = np.arange(1, q)
        exact = 1 + 2 * np.sum((1 - k / q) * phi**k)
        se = np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
        assert abs(vr - exact) < 4 * se


def test_robust_z_equals_homoskedastic_z_for_iid_increments():
    """For i.i.d. increments delta(j) -> 1, so theta(q) -> 2(2q-1)(q-1)/(3q) and z* -> z."""
    p = np.cumsum(np.random.default_rng(9).standard_normal(200_001))
    res = variance_ratio(p, q=[2, 8])
    np.testing.assert_allclose(res.z_robust, res.z, rtol=0.05)


def test_variance_ratio_size_and_heteroskedasticity_robustness():
    rng = np.random.default_rng(7)
    z = [variance_ratio(np.cumsum(rng.standard_normal(1001)), q=4).z[0] for _ in range(400)]
    assert _rejection_rate_ok(np.abs(z) > 1.96, 400)
    # Under GARCH increments the homoskedastic z is too dispersed; the robust z* has
    # variance close to one, since its standard error accounts for volatility clustering.
    m = GARCH(omega=0.02, alpha=0.2, beta=0.78)
    zr, zh = [], []
    for _ in range(400):
        res = variance_ratio(np.cumsum(m.simulate(2000, burn=200, rng=rng).returns), q=2)
        zh.append(res.z[0])
        zr.append(res.z_robust[0])
    assert np.var(zh) > 1 + 4 * np.sqrt(2 / 400)
    assert abs(np.var(zr) - 1) < 4 * np.sqrt(2 / 400)


def test_variance_ratio_validation_and_pvalues():
    p = np.cumsum(np.random.default_rng(8).standard_normal(101))
    res = variance_ratio(p, q=[2, 5])
    assert res.p_value.shape == (2,) and np.all((res.p_value_robust > 0) & (res.p_value_robust <= 1))
    with pytest.raises(ValueError):
        variance_ratio(p, q=1)


def test_runs_test_needs_three_observations():
    with pytest.raises(ValueError, match="at least three"):
        runs_test([1.0, -1.0])
