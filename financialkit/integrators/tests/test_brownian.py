"""Tests for the Brownian increment generators and the Brownian bridge."""

import numpy as np
import pytest

from financialkit.integrators import (
    antithetic_increments,
    brownian_bridge,
    brownian_increments,
    correlated_increments,
    sobol_increments,
)
from financialkit.integrators.brownian import _bridge_order


def test_brownian_increments_shape_and_seed():
    assert brownian_increments(1.0, 10, 7, rng=0).shape == (7, 10)
    assert brownian_increments(1.0, 10, 7, dim=3, rng=0).shape == (7, 10, 3)
    np.testing.assert_array_equal(brownian_increments(1.0, 4, 3, rng=5), brownian_increments(1.0, 4, 3, rng=5))


def test_brownian_increments_terminal_value_is_n_0_T():
    """W_T = sum dW ~ N(0, T): mean within 4 standard errors of 0, variance
    within 4 standard errors (T sqrt(2 / n)) of T."""
    T, n = 2.0, 50_000
    W_T = brownian_increments(T, 16, n, rng=1).sum(axis=1)
    assert abs(W_T.mean()) < 4 * np.sqrt(T / n)
    assert abs(W_T.var(ddof=1) - T) < 4 * T * np.sqrt(2 / n)


@pytest.mark.parametrize(("T", "n_steps", "n_paths"), [(0.0, 4, 4), (1.0, 0, 4), (1.0, 4, 0), (1.0, 2.5, 4)])
def test_invalid_grid_raises(T, n_steps, n_paths):
    with pytest.raises(ValueError):
        brownian_increments(T, n_steps, n_paths)


def test_correlated_increments_have_target_correlation():
    """Sample correlation of n pairs has standard error ~ (1 - rho^2) / sqrt(n)."""
    rho, n = 0.6, 40_000
    dW = correlated_increments([[1.0, rho], [rho, 1.0]], T=0.5, n_steps=2, n_paths=n, rng=2)
    assert dW.shape == (n, 2, 2)
    r = np.corrcoef(dW[:, :, 0].ravel(), dW[:, :, 1].ravel())[0, 1]
    assert abs(r - rho) < 4 * (1 - rho**2) / np.sqrt(2 * n)
    assert abs(dW[:, :, 0].var() - 0.25) < 4 * 0.25 * np.sqrt(2 / (2 * n))


@pytest.mark.parametrize(
    "corr",
    [
        [[1.0, 0.5], [0.4, 1.0]],  # not symmetric
        [[2.0, 0.0], [0.0, 1.0]],  # not unit diagonal
        [[1.0, 1.5], [1.5, 1.0]],  # not positive definite
        [1.0, 0.5],  # not a matrix
    ],
)
def test_correlated_increments_rejects_invalid_matrix(corr):
    with pytest.raises(ValueError):
        correlated_increments(corr, 1.0, 2, 2)


def test_antithetic_increments_pair_each_path_with_its_negative():
    dW = antithetic_increments(1.0, 5, 8, dim=2, rng=3)
    np.testing.assert_array_equal(dW[4:], -dW[:4])
    with pytest.raises(ValueError, match="even"):
        antithetic_increments(1.0, 5, 7)


@pytest.mark.parametrize("bridge", [True, False])
def test_sobol_increments_reproduce_brownian_moments(bridge):
    """Scrambled Sobol points with 4096 paths fix E[W_T] = 0 and
    E[W_T^2] = T far more tightly than pseudo-random draws would (whose
    standard errors here are ~0.016 and ~0.022)."""
    T = 1.0
    dW = sobol_increments(T, 8, 4096, bridge=bridge, rng=4)
    assert dW.shape == (4096, 8)
    W_T = dW.sum(axis=1)
    assert abs(W_T.mean()) < 0.005
    assert abs((W_T**2).mean() - T) < 0.01
    np.testing.assert_array_equal(dW, sobol_increments(T, 8, 4096, bridge=bridge, rng=4))


@pytest.mark.parametrize("n_steps", [1, 2, 5, 8, 13])
def test_bridge_order_fills_every_interior_point_once(n_steps):
    mids = [m for _, m, _ in _bridge_order(n_steps)]
    assert sorted(mids) == list(range(1, n_steps))
    known = {0, n_steps}
    for l, m, r in _bridge_order(n_steps):
        assert l in known and r in known and l < m < r
        known.add(m)


def test_brownian_bridge_terminal_value_comes_from_first_normal():
    z = np.random.default_rng(5).standard_normal((3, 6))
    W = brownian_bridge(z, T=4.0)
    assert W.shape == (3, 7)
    np.testing.assert_array_equal(W[:, 0], 0.0)
    np.testing.assert_allclose(W[:, -1], 2.0 * z[:, 0])


def test_brownian_bridge_has_brownian_covariance():
    """cov(W_s, W_t) = min(s, t); each sample covariance of n paths has a
    standard error below sqrt(2) T / sqrt(n)."""
    T, n_steps, n = 1.5, 6, 60_000
    W = brownian_bridge(np.random.default_rng(6).standard_normal((n, n_steps)), T)[:, 1:]
    t = np.linspace(0, T, n_steps + 1)[1:]
    np.testing.assert_allclose(np.cov(W, rowvar=False), np.minimum.outer(t, t), atol=4 * np.sqrt(2) * T / np.sqrt(n))
