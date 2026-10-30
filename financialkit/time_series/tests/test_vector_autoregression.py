"""Tests for Sims's VAR: estimation, impulse responses, variance decompositions and forecasts."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from financialkit.time_series import fit_var, ols, simulate_var

A1 = np.array([[0.5, 0.2, 0.0], [0.1, 0.4, -0.2], [0.0, 0.3, 0.6]])
A2 = np.array([[-0.2, 0.0, 0.1], [0.0, 0.1, 0.0], [0.1, 0.0, -0.1]])
SIGMA = np.array([[1.0, 0.3, 0.1], [0.3, 0.5, 0.0], [0.1, 0.0, 0.8]])


def test_var_is_equation_by_equation_least_squares_and_recovers_the_coefficients():
    y = simulate_var([A1, A2], 3000, sigma=SIGMA, intercept=[0.1, 0.0, -0.1], rng=0)
    res = fit_var(y, 2)
    X = np.column_stack([np.ones(2998), y[1:-1], y[:-2]])
    for a in range(3):
        eq = ols(y[2:, a], X)
        np.testing.assert_allclose(res.intercept[a], eq.coef[0], rtol=1e-10)
        np.testing.assert_allclose(res.coefs[:, a, :].ravel(), eq.coef[1:], rtol=1e-9, atol=1e-12)
        true = np.r_[A1[a], A2[a]]
        assert np.all(np.abs(eq.coef[1:] - true) < 4 * eq.std_errors[1:])
    assert res.is_stable() and res.order == 2 and res.n_obs == 2998 and res.n_params == 21


def test_loglik_is_the_gaussian_density_of_the_residuals():
    y = simulate_var(A1, 400, sigma=SIGMA, rng=1)
    res = fit_var(y, 1)
    assert res.loglik == pytest.approx(multivariate_normal(np.zeros(3), res.sigma).logpdf(res.residuals).sum(), rel=1e-10)
    assert res.bic > res.aic


def test_impulse_responses_and_fevd():
    res = fit_var(simulate_var(A1, 500, sigma=SIGMA, rng=2), 1)
    A, P = res.coefs[0], np.linalg.cholesky(res.sigma)
    irf = res.impulse_response(5, orthogonalized=False)
    for h in range(6):
        np.testing.assert_allclose(irf[h], np.linalg.matrix_power(A, h), atol=1e-12)
    np.testing.assert_allclose(res.impulse_response(5)[0], P, atol=1e-12)
    fevd = res.fevd(8)
    np.testing.assert_allclose(fevd.sum(axis=2), 1.0, rtol=1e-12)
    np.testing.assert_allclose(fevd[0], P**2 / (P**2).sum(axis=1, keepdims=True), rtol=1e-12)
    assert fevd[0, 0, 1] == 0.0  # the first variable is ordered first: no same-period response to later shocks


def test_ma_weights_are_the_companion_powers_and_forecasts_mean_revert():
    res = fit_var(simulate_var([A1, A2], 800, rng=3), 2)
    C = res.companion()
    Phi = res.ma_weights(6)
    for h in range(7):
        np.testing.assert_allclose(Phi[h], np.linalg.matrix_power(C, h)[:3, :3], atol=1e-12)
    res1 = fit_var(simulate_var(A1, 800, intercept=[1.0, 0.0, 0.0], rng=4), 1)
    mu = np.linalg.solve(np.eye(3) - res1.coefs[0], res1.intercept)
    f = res1.forecast(4)
    for h in range(1, 5):
        np.testing.assert_allclose(f[h - 1], mu + np.linalg.matrix_power(res1.coefs[0], h) @ (res1.data[-1] - mu), rtol=1e-10)


def test_input_checks():
    y = simulate_var(A1, 50, rng=5)
    with pytest.raises(ValueError):
        fit_var(y[:, 0], 1)
    with pytest.raises(ValueError):
        fit_var(y, 2, start=1)
    with pytest.raises(ValueError):
        fit_var(y[:5], 2)
    with pytest.raises(ValueError):
        simulate_var(np.ones((2, 3)), 10)
    assert np.all(fit_var(y, 1, include_mean=False).intercept == 0.0)
