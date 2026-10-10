"""Tests for the SDE schemes, the exact samplers and the convergence-order estimators."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_ATOL, DEFAULT_RTOL
from financialkit.integrators import (
    ConvergenceResult,
    brownian_increments,
    correlated_increments,
    euler_maruyama,
    gbm_paths,
    milstein,
    njit,
    ou_paths,
    strong_convergence,
    weak_convergence,
)


@njit
def _linear(x, t, p):
    return p[0] * x


@njit
def _scaled(x, t, p):
    return p[1] * x


@njit
def _slope(x, t, p):
    return p[1] + 0.0 * x


@njit
def _const_drift(x, t, p):
    return p[0] + 0.0 * x


@njit
def _zero(x, t, p):
    return 0.0 * x


@njit
def _time(x, t, p):
    return t + 0.0 * x


def _gbm_exact(mu, sigma, T):
    return lambda W: np.exp((mu - 0.5 * sigma**2) * T + sigma * W)


def test_euler_maruyama_without_noise_is_explicit_euler():
    """dx = -x dt: Euler gives x_n = (1 - dt)^n exactly."""
    times, paths = euler_maruyama(_linear, _zero, 1.0, T=1.0, n_steps=10, n_paths=3, params=[-1.0, 0.0], rng=0)
    np.testing.assert_allclose(times, np.linspace(0, 1, 11))
    np.testing.assert_allclose(paths, np.broadcast_to(0.9 ** np.arange(11), (3, 11)), rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_euler_maruyama_is_exact_for_arithmetic_brownian_motion():
    """dX = mu dt + sigma dW has constant coefficients, so each Euler step is exact:
    X_t = x0 + mu t + sigma W_t on the same increments."""
    mu, sigma = 0.3, 0.7
    dW = brownian_increments(2.0, 8, 5, rng=1)
    times, paths = euler_maruyama(_const_drift, _slope, 1.0, T=2.0, n_steps=8, n_paths=5, params=[mu, sigma], dW=dW)
    W = np.concatenate([np.zeros((5, 1)), np.cumsum(dW, axis=1)], axis=1)
    np.testing.assert_allclose(paths, 1.0 + mu * times + sigma * W, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_euler_maruyama_evaluates_drift_at_left_endpoints_from_t0():
    """dx = t dt from t0: Euler is the left Riemann sum of t over the grid."""
    t0, T, n = 0.5, 1.0, 4
    times, paths = euler_maruyama(_time, _zero, 0.0, T=T, n_steps=n, n_paths=1, t0=t0)
    assert times[0] == t0 and times[-1] == pytest.approx(t0 + T)
    assert paths[0, -1] == pytest.approx(np.sum(times[:-1]) * T / n, rel=DEFAULT_RTOL)


def test_milstein_with_constant_diffusion_equals_euler_maruyama():
    dW = brownian_increments(1.0, 6, 4, rng=2)
    _, em = euler_maruyama(_const_drift, _slope, 2.0, 1.0, 6, 4, params=[0.1, 0.4], dW=dW)
    _, mil = milstein(_const_drift, _slope, _zero, 2.0, 1.0, 6, 4, params=[0.1, 0.4], dW=dW)
    np.testing.assert_allclose(mil, em, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_schemes_track_exact_gbm_on_shared_increments():
    """On a fine grid both schemes stay pathwise close to the exact solution
    driven by the same Brownian motion, Milstein much closer than Euler."""
    s0, mu, sigma, T, n = 100.0, 0.05, 0.3, 1.0, 512
    dW = brownian_increments(T, n, 200, rng=3)
    _, exact = gbm_paths(s0, mu, sigma, T, n, 200, dW=dW)
    _, em = euler_maruyama(_linear, _scaled, s0, T, n, 200, params=[mu, sigma], dW=dW)
    _, mil = milstein(_linear, _scaled, _slope, s0, T, n, 200, params=[mu, sigma], dW=dW)
    em_err, mil_err = np.abs(em - exact).max(axis=1).mean(), np.abs(mil - exact).max(axis=1).mean()
    assert em_err < 1.0
    assert mil_err < em_err / 5


def test_vector_state_with_correlated_drivers():
    """Two GBMs with correlated drivers: shapes, and each component matches
    its own exact solution as the step shrinks."""
    corr = [[1.0, -0.5], [-0.5, 1.0]]
    dW = correlated_increments(corr, 1.0, 256, 100, rng=4)
    times, paths = milstein(_linear, _scaled, _slope, [1.0, 2.0], 1.0, 256, 100, params=[0.1, 0.2], dW=dW)
    assert paths.shape == (100, 257, 2)
    for k, s0 in enumerate([1.0, 2.0]):
        _, exact = gbm_paths(s0, 0.1, 0.2, 1.0, 256, 100, dW=dW[:, :, k])
        np.testing.assert_allclose(paths[:, :, k], exact, rtol=1e-3)


def test_schemes_validate_inputs():
    with pytest.raises(ValueError, match="dW must have shape"):
        euler_maruyama(_linear, _scaled, 1.0, 1.0, 4, 3, params=[0.0, 0.1], dW=np.zeros((3, 5)))
    with pytest.raises(ValueError, match="dW must have shape"):
        euler_maruyama(_linear, _scaled, [1.0, 1.0], 1.0, 4, 3, params=[0.0, 0.1], dW=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="x0"):
        euler_maruyama(_linear, _scaled, np.ones((2, 2)), 1.0, 4, 3, params=[0.0, 0.1])


def test_gbm_paths_log_moments():
    """log S_T ~ N(log s0 + (mu - sigma^2/2) T, sigma^2 T), checked within 4 standard errors."""
    s0, mu, sigma, T, n = 50.0, 0.08, 0.25, 2.0, 50_000
    times, S = gbm_paths(s0, mu, sigma, T, 4, n, rng=5)
    assert S.shape == (n, 5) and np.all(S[:, 0] == s0)
    x = np.log(S[:, -1])
    v = sigma**2 * T
    assert abs(x.mean() - (np.log(s0) + (mu - 0.5 * sigma**2) * T)) < 4 * np.sqrt(v / n)
    assert abs(x.var(ddof=1) - v) < 4 * v * np.sqrt(2 / n)
    with pytest.raises(ValueError, match="dW must have shape"):
        gbm_paths(s0, mu, sigma, T, 4, n, dW=np.zeros((n, 3)))


@pytest.mark.parametrize("kappa", [0.0, 1.5])
def test_ou_paths_terminal_moments(kappa):
    """X_T ~ N(theta + (x0 - theta) e^{-kappa T}, sigma^2 (1 - e^{-2 kappa T}) / (2 kappa)),
    variance sigma^2 T when kappa = 0; checked within 4 standard errors."""
    x0, theta, sigma, T, n = 0.08, 0.03, 0.02, 3.0, 50_000
    _, X = ou_paths(x0, kappa, theta, sigma, T, 6, n, rng=6)
    mean = theta + (x0 - theta) * np.exp(-kappa * T)
    var = sigma**2 * T if kappa == 0 else sigma**2 * -np.expm1(-2 * kappa * T) / (2 * kappa)
    assert abs(X[:, -1].mean() - mean) < 4 * np.sqrt(var / n)
    assert abs(X[:, -1].var(ddof=1) - var) < 4 * var * np.sqrt(2 / n)


@pytest.mark.parametrize(("scheme", "expected"), [("euler_maruyama", 0.5), ("milstein", 1.0)])
def test_strong_order_on_gbm(scheme, expected):
    """Kloeden-Platen (1992), Thms. 10.2.2 and 10.3.5: strong orders 1/2 and 1."""
    mu, sigma = 1.0, 0.5
    res = strong_convergence(_linear, _scaled, 1.0, 1.0, _gbm_exact(mu, sigma, 1.0), params=[mu, sigma], scheme=scheme, diffusion_dx=_slope, rng=7)
    assert isinstance(res, ConvergenceResult)
    assert np.all(np.diff(res.errors) < 0)
    assert res.order == pytest.approx(expected, abs=0.15)


def test_strong_order_vector_state():
    mu, sigma = 1.0, 0.5
    exact = lambda W: np.exp((mu - 0.5 * sigma**2) + sigma * W)
    res = strong_convergence(_linear, _scaled, [1.0, 1.0], 1.0, exact, params=[mu, sigma], rng=8)
    assert res.order == pytest.approx(0.5, abs=0.15)


def test_weak_order_of_euler_maruyama_on_gbm_mean():
    """E[X_T] = e^{mu T} while Euler gives (1 + mu dt)^n: weak order 1
    (Kloeden-Platen 1992, Thm. 14.5.1). The bias dwarfs the Monte Carlo noise."""
    mu, sigma = 1.0, 0.2
    res = weak_convergence(_linear, _scaled, 1.0, 1.0, np.exp(mu), params=[mu, sigma], rng=9)
    assert np.all(res.errors > 10 * res.standard_errors)
    assert res.order == pytest.approx(1.0, abs=0.15)
    payoff = weak_convergence(_linear, _scaled, 1.0, 1.0, np.exp(2 * mu + sigma**2), payoff=np.square, params=[mu, sigma], rng=9)
    assert payoff.order == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_steps_list": [8]}, "at least two"),
        ({"n_steps_list": [3, 8]}, "divide"),
        ({"scheme": "milstein"}, "diffusion_dx"),
        ({"scheme": "rk4"}, "scheme"),
    ],
)
def test_convergence_validates_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        strong_convergence(_linear, _scaled, 1.0, 1.0, lambda W: W, params=[0.0, 0.1], **kwargs)
