"""Tests for the Thomas algorithm, the tridiagonal helpers and the Black-Scholes operator."""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from financialkit.constants import DEFAULT_ATOL, DEFAULT_RTOL
from financialkit.pde_methods import (
    black_scholes_operator,
    solve_tridiagonal,
    tavella_randall_grid,
    thomas_solve,
    tridiagonal_matrix,
)


def _random_system(m, seed=0):
    rng = np.random.default_rng(seed)
    lower, upper = rng.uniform(-1, 1, m), rng.uniform(-1, 1, m)
    diag = 2.5 + rng.uniform(0, 1, m)  # diagonally dominant, so no pivoting is needed
    return lower, diag, upper, rng.normal(size=m)


def test_thomas_matches_sparse_direct_solve():
    lower, diag, upper, rhs = _random_system(50)
    x = thomas_solve(lower, diag, upper, rhs)
    np.testing.assert_allclose(x, spsolve(tridiagonal_matrix(lower, diag, upper).tocsc(), rhs), rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_solve_tridiagonal_banded_and_thomas_agree():
    lower, diag, upper, rhs = _random_system(40, seed=1)
    a = solve_tridiagonal(lower, diag, upper, rhs, "banded")
    b = solve_tridiagonal(lower, diag, upper, rhs, "thomas")
    np.testing.assert_allclose(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_solve_tridiagonal_many_right_hand_sides():
    lower, diag, upper, _ = _random_system(30, seed=2)
    rhs = np.random.default_rng(3).normal(size=(30, 4))
    X = solve_tridiagonal(lower, diag, upper, rhs)
    np.testing.assert_allclose(tridiagonal_matrix(lower, diag, upper) @ X, rhs, atol=1e-10)


def test_solver_validation():
    with pytest.raises(ValueError):
        thomas_solve([0, 1], [1, 1, 1], [1, 0], [1, 1, 1])
    with pytest.raises(ValueError):
        solve_tridiagonal([0, 1], [1, 1], [1, 0], [1, 1], solver="lu")


def test_operator_on_uniform_grid_is_wilmotts():
    """Wilmott, *Paul Wilmott on Quantitative Finance*, 2nd ed., Ch. 78: coefficients (sigma^2 i^2 -+ (r - q) i)/2 and -(sigma^2 i^2 + r)."""
    sigma, r, q = 0.3, 0.05, 0.02
    i = np.arange(1, 20)
    lower, diag, upper = black_scholes_operator(np.arange(21.0) * 2.5, sigma, r, q)
    np.testing.assert_allclose(lower, 0.5 * (sigma**2 * i**2 - (r - q) * i), rtol=DEFAULT_RTOL)
    np.testing.assert_allclose(upper, 0.5 * (sigma**2 * i**2 + (r - q) * i), rtol=DEFAULT_RTOL)
    np.testing.assert_allclose(diag, -(sigma**2 * i**2 + r), rtol=DEFAULT_RTOL)


def test_operator_is_exact_for_quadratics_on_a_nonuniform_grid():
    # three-point differences differentiate quadratics exactly on any grid
    S = tavella_randall_grid(0.0, 300.0, 60, 100.0, 15.0)
    sigma, r, q = 0.25, 0.04, 0.01
    lower, diag, upper = black_scholes_operator(S, sigma, r, q)
    V = 3.0 + 0.5 * S + 0.01 * S**2
    LV = lower * V[:-2] + diag * V[1:-1] + upper * V[2:]
    s = S[1:-1]
    exact = 0.5 * sigma**2 * s**2 * 0.02 + (r - q) * s * (0.5 + 0.02 * s) - r * (3.0 + 0.5 * s + 0.01 * s**2)
    np.testing.assert_allclose(LV, exact, rtol=1e-9, atol=1e-9)


def test_operator_rejects_bad_grids():
    with pytest.raises(ValueError):
        black_scholes_operator([0.0, 2.0, 1.0], 0.2, 0.05)
