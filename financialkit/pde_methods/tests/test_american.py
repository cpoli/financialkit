"""Tests for PSOR, the Brennan-Schwartz algorithm, the penalty method and American options."""

import numpy as np
import pytest

from financialkit.options import CRRTree, black_scholes_price
from financialkit.pde_methods import CrankNicolson, ExplicitScheme, ImplicitScheme, brennan_schwartz, penalty_solve, psor, tridiagonal_matrix


def _lcp(m=40, seed=0):
    """A put-like LCP: an M-matrix and an obstacle that binds at the low end."""
    rng = np.random.default_rng(seed)
    lower, upper = -rng.uniform(0.2, 0.5, m), -rng.uniform(0.2, 0.5, m)
    diag = 1.2 + rng.uniform(0, 0.3, m)
    rhs = np.linspace(0.2, 1.0, m)
    obstacle = np.maximum(1.5 - np.linspace(0, 3, m), 0.0)
    return lower, diag, upper, rhs, obstacle


def _check_complementarity(lower, diag, upper, rhs, obstacle, x, tol):
    residual = tridiagonal_matrix(lower, diag, upper) @ x - rhs
    assert np.all(x >= obstacle - tol)
    assert np.all(residual >= -tol)
    assert np.max(np.abs(residual * (x - obstacle))) < tol


def test_lcp_solvers_agree_and_satisfy_complementarity():
    lower, diag, upper, rhs, obstacle = _lcp()
    a = psor(lower, diag, upper, rhs, obstacle, omega=1.3, tol=1e-12)
    b = brennan_schwartz(lower, diag, upper, rhs, obstacle, exercise="low")
    c = penalty_solve(lower, diag, upper, rhs, obstacle)
    assert b.iterations == 1 and a.iterations > 1 and c.iterations < 10
    np.testing.assert_allclose(a.x, b.x, atol=1e-10)
    np.testing.assert_allclose(c.x, b.x, atol=1e-7)  # penalty error O(1 / rho) with rho = 1e8
    _check_complementarity(lower, diag, upper, rhs, obstacle, b.x, 1e-10)
    assert np.any(b.x == obstacle) and np.any(b.x > obstacle)


def test_brennan_schwartz_high_side_matches_psor():
    lower, diag, upper, rhs, obstacle = _lcp(seed=1)
    obstacle = obstacle[::-1].copy()
    a = psor(lower, diag, upper, rhs, obstacle, tol=1e-12)
    b = brennan_schwartz(lower, diag, upper, rhs, obstacle, exercise="high")
    np.testing.assert_allclose(a.x, b.x, atol=1e-10)


def test_lcp_solvers_validate_and_report_failure():
    lower, diag, upper, rhs, obstacle = _lcp()
    with pytest.raises(RuntimeError):
        psor(lower, diag, upper, rhs, obstacle, max_iter=2)
    with pytest.raises(ValueError):
        psor(lower, diag, upper, rhs, obstacle, omega=2.5)
    with pytest.raises(ValueError):
        brennan_schwartz(lower, diag, upper, rhs, obstacle, exercise="middle")
    with pytest.raises(ValueError):
        psor(lower[:-1], diag, upper, rhs, obstacle)
    with pytest.raises(RuntimeError):
        penalty_solve(lower, diag, upper, rhs, obstacle, max_iter=0)


@pytest.mark.parametrize("method", ["psor", "brennan_schwartz", "penalty"])
def test_american_put_hull_example_21_1(method):
    """Hull, 10th ed., Example 21.1 (S = K = 50, r = 10%, sigma = 40%, five months): 4.28 by a fine tree."""
    reference = CRRTree(5000).price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True).price
    res = CrankNicolson(400, 200, rannacher_steps=2, american_method=method).price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True)
    # grid error O(h^2) with h = 0.5, plus the tree's own O(1/n) error
    assert res.price == pytest.approx(reference, abs=3e-3)
    assert res.price > black_scholes_price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put")
    b = res.exercise_boundary
    assert b.shape == (201,) and np.all(np.isfinite(b))
    assert np.all(np.diff(b) >= -1e-12) and b[-1] == pytest.approx(50.0, abs=0.5)  # rises to the strike at expiry
    assert np.all(res.V >= np.maximum(50.0 - res.S, 0.0) - 1e-7)


def test_american_methods_agree_closely():
    args = (100.0, 100.0, 1.0, 0.06, 0.3, 0.0, "put", True)
    prices = [CrankNicolson(200, 100, rannacher_steps=2, american_method=m).price(*args).price for m in ("psor", "brennan_schwartz", "penalty")]
    assert np.ptp(prices) < 1e-6


def test_explicit_and_implicit_american_puts():
    reference = CRRTree(5000).price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True).price
    ex = ExplicitScheme(200, 3000).price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True)
    im = ImplicitScheme(400, 800, american_method="brennan_schwartz").price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True)
    assert ex.stability.stable
    assert ex.price == pytest.approx(reference, abs=0.02)
    assert im.price == pytest.approx(reference, abs=0.01)


def test_american_call_without_dividends_is_european():
    eu = CrankNicolson(200, 100, rannacher_steps=2).price(100.0, 100.0, 1.0, 0.05, 0.2)
    am = CrankNicolson(200, 100, rannacher_steps=2, american_method="brennan_schwartz").price(100.0, 100.0, 1.0, 0.05, 0.2, american=True)
    assert am.price == pytest.approx(eu.price, abs=1e-8)


def test_american_call_with_dividends_matches_tree():
    reference = CRRTree(4000).price(100.0, 100.0, 1.0, 0.05, 0.2, q=0.08, option_type="call", american=True).price
    res = CrankNicolson(800, 200, rannacher_steps=2, american_method="brennan_schwartz").price(100.0, 100.0, 1.0, 0.05, 0.2, q=0.08, american=True)
    assert res.price == pytest.approx(reference, abs=5e-3)
    assert np.nanmin(res.exercise_boundary) > 100.0
