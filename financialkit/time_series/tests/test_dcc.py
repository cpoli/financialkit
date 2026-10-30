"""Tests for DCC-GARCH correlations, likelihood, estimation and simulation."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from financialkit.time_series import GARCH, dcc_correlations, dcc_loglikelihood, fit_dcc, simulate_dcc

G = GARCH(omega=0.02, alpha=0.08, beta=0.9)
RBAR = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.5], [0.2, 0.5, 1.0]])


def test_correlations_match_the_explicit_recursion():
    z = np.random.default_rng(0).standard_normal((60, 3))
    a, b = 0.07, 0.9
    Qbar = z.T @ z / 60
    R = dcc_correlations(z, a, b)
    Q = Qbar.copy()
    for t in range(60):
        d = 1 / np.sqrt(np.diag(Q))
        np.testing.assert_allclose(R[t], d[:, None] * Q * d[None, :], rtol=1e-10)
        Q = (1 - a - b) * Qbar + a * np.outer(z[t], z[t]) + b * Q
    np.testing.assert_allclose(dcc_correlations(z, 0.0, 0.0, RBAR), np.broadcast_to(RBAR, (60, 3, 3)), atol=1e-12)


def test_correlation_loglik_is_the_density_ratio():
    z = np.random.default_rng(1).standard_normal((40, 3))
    R = dcc_correlations(z, 0.05, 0.9)
    expected = sum(multivariate_normal(np.zeros(3), R[t]).logpdf(z[t]) for t in range(40)) - multivariate_normal(np.zeros(3), np.eye(3)).logpdf(z).sum()
    assert dcc_loglikelihood(z, 0.05, 0.9) == pytest.approx(expected, rel=1e-10)


def test_simulation_has_the_target_correlation():
    r, R = simulate_dcc([G, G, G], 0.04, 0.94, RBAR, n=5000, rng=2)
    assert r.shape == (5000, 3) and R.shape == (5000, 3, 3)
    np.testing.assert_allclose(R[:, [0, 1, 2], [0, 1, 2]], 1.0)
    assert np.all(np.abs(R[:, 0, 1]) < 1)
    # The time-averaged conditional correlation stays close to Rbar.
    np.testing.assert_allclose(R.mean(axis=0), RBAR, atol=0.05)


def test_two_step_fit_recovers_the_dynamics():
    r, R = simulate_dcc([G, G], 0.05, 0.93, RBAR[:2, :2], n=5000, rng=3)
    fit = fit_dcc(r)
    assert fit.converged
    assert abs(fit.a - 0.05) < 0.02 and abs(fit.b - 0.93) < 0.03
    assert fit.persistence < 1
    assert np.mean(np.abs(fit.correlation[:, 0, 1] - R[:, 0, 1])) < 0.08
    np.testing.assert_allclose(fit.covariance[:, 0, 0], fit.univariate[0].variance, rtol=1e-12)
    assert fit.loglik == pytest.approx(fit.univariate[0].loglik + fit.univariate[1].loglik + fit.correlation_loglik)
    # Dynamic correlations beat constant ones in likelihood on DCC data.
    assert fit.correlation_loglik > dcc_loglikelihood(fit.standardized_residuals, 0.0, 0.0, fit.Qbar)
    with pytest.raises(ValueError):
        fit_dcc(r[:, 0])


def test_indefinite_target_gives_minus_infinite_loglik():
    z = np.random.default_rng(4).standard_normal((10, 2))
    assert dcc_loglikelihood(z, 0.0, 0.0, Qbar=[[1.0, 2.0], [2.0, 1.0]]) == -np.inf
