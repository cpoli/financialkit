"""Tests for the Kalman filter, the RTS smoother and the local-level model."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from financialkit.time_series import fit_local_level, kalman_filter, local_level_steady_state_gain


def test_loglik_is_the_exact_gaussian_likelihood_of_ar1_plus_noise():
    """The prediction-error decomposition equals the joint normal density of y."""
    phi, q, r, n = 0.8, 0.5, 0.3, 40
    y = np.random.default_rng(0).standard_normal(n)
    res = kalman_filter(y, F=phi, H=1.0, Q=q, R=r)  # stationary P0 = q / (1 - phi^2)
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    cov = q / (1 - phi**2) * phi**lag + r * np.eye(n)
    assert res.loglik == pytest.approx(multivariate_normal(np.zeros(n), cov).logpdf(y), rel=1e-10)
    assert res.n_obs == n


def _random_walk_posterior(y, q, r, P0, keep):
    """E[x | observed y] and Var for x_t = x_{t-1} + w_t, x_0 ~ N(0, P0), by direct Gaussian conditioning."""
    n = y.size
    t = np.arange(1, n + 1)
    Sxx = P0 + q * np.minimum.outer(t, t)
    Syy = Sxx[np.ix_(keep, keep)] + r * np.eye(keep.size)
    Sxy = Sxx[:, keep]
    mean = Sxy @ np.linalg.solve(Syy, y[keep])
    var = np.diag(Sxx - Sxy @ np.linalg.solve(Syy, Sxy.T))
    return mean, var


def test_smoother_and_filter_match_direct_conditioning_with_a_missing_value():
    q, r, P0, n = 0.4, 1.0, 2.0, 30
    rng = np.random.default_rng(1)
    y = np.cumsum(np.sqrt(q) * rng.standard_normal(n)) + rng.standard_normal(n)
    y[10] = np.nan
    keep = np.flatnonzero(~np.isnan(y))
    res = kalman_filter(y, F=1.0, H=1.0, Q=q, R=r, x0=0.0, P0=P0)
    xs, Ps = res.smooth()
    mean, var = _random_walk_posterior(y, q, r, P0, keep)
    np.testing.assert_allclose(xs[:, 0], mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(Ps[:, 0, 0], var, rtol=1e-9)
    # The filtered state at the end uses all the data, so it equals the smoothed one.
    assert res.filtered_state[-1, 0] == pytest.approx(mean[-1], rel=1e-9)
    # The missing date is a pure prediction.
    assert res.n_obs == n - 1 and np.isnan(res.innovations[10, 0])
    assert res.filtered_state[10, 0] == res.predicted_state[10, 0]


@pytest.mark.parametrize("q", [0.01, 1.0, 25.0])
def test_gain_converges_to_the_riccati_steady_state(q):
    res = kalman_filter(np.zeros(300), F=1.0, H=1.0, Q=q, R=1.0, x0=0.0, P0=100.0)
    assert res.gain[-1, 0, 0] == pytest.approx(local_level_steady_state_gain(q), rel=1e-10)


def test_bivariate_observations_and_shape_checks():
    F = np.array([[1.0, 1.0], [0.0, 1.0]])  # local linear trend
    H = np.eye(2)
    y = np.random.default_rng(2).standard_normal((50, 2))
    res = kalman_filter(y, F, H, Q=0.1 * np.eye(2), R=np.eye(2), P0=10 * np.eye(2))
    assert res.filtered_cov.shape == (50, 2, 2) and np.isfinite(res.loglik)
    with pytest.raises(ValueError):
        kalman_filter(y, F, np.eye(3), Q=np.eye(2), R=np.eye(2))
    with pytest.raises(ValueError):
        kalman_filter(y, np.ones((2, 2, 2)), H, Q=np.eye(2), R=np.eye(2))


def test_local_level_maximum_likelihood_recovers_the_variances():
    rng = np.random.default_rng(3)
    n, s_eta, s_eps = 4000, 0.3, 1.2
    y = np.cumsum(np.sqrt(s_eta) * rng.standard_normal(n)) + np.sqrt(s_eps) * rng.standard_normal(n)
    fit = fit_local_level(y)
    assert fit.converged
    # Asymptotic standard errors at n = 4000 are under 10% of each variance.
    assert fit.sigma2_level == pytest.approx(s_eta, rel=0.25)
    assert fit.sigma2_noise == pytest.approx(s_eps, rel=0.15)
    assert fit.signal_to_noise == pytest.approx(fit.sigma2_level / fit.sigma2_noise)
    assert fit.filter.gain[-1, 0, 0] == pytest.approx(fit.steady_state_gain, rel=1e-6)
    with pytest.raises(ValueError):
        fit_local_level([1.0, 2.0])
    with pytest.raises(ValueError):
        fit_local_level(np.arange(10.0))
