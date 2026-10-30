"""Tests for financialkit.exotic_options.systems.american."""

import numpy as np
import pytest

from financialkit.exotic_options import longstaff_schwartz, regression_basis
from financialkit.options import CRRTree, black_scholes_price


@pytest.mark.parametrize(("S", "T", "fd"), [(36.0, 1.0, 4.478), (36.0, 2.0, 4.840), (40.0, 1.0, 2.314), (44.0, 1.0, 1.110)])
def test_longstaff_schwartz_table_1(S, T, fd):
    """Longstaff and Schwartz (2001), Table 1 (K = 40, r = 6%, sigma = 20%, 50 exercise dates a year):
    finite-difference values. The regression policy is slightly suboptimal and fitted in-sample,
    so allow two cents of bias beyond the sampling noise."""
    res = longstaff_schwartz(S, 40.0, T, 0.06, 0.2, n_steps=int(50 * T), n_paths=40_000, rng=31)
    assert abs(res.estimate - fd) < 4 * res.std_error + 0.02


def test_american_put_between_european_and_tree():
    res = longstaff_schwartz(100.0, 100.0, 1.0, 0.05, 0.3, n_paths=40_000, basis="monomial", rng=32)
    european = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.3, option_type="put")
    tree = CRRTree(1000).price(100.0, 100.0, 1.0, 0.05, 0.3, option_type="put", american=True).price
    assert res.estimate > european + 4 * res.std_error
    assert abs(res.estimate - tree) < 4 * res.std_error + 0.03
    assert 0.0 < res.exercise_probability < 1.0
    finite = res.exercise_boundary[np.isfinite(res.exercise_boundary)]
    assert finite.size > 10 and np.all(finite <= 100.0)


def test_call_without_dividends_is_european():
    """Merton (1973): never exercise an American call on a non-dividend stock early."""
    res = longstaff_schwartz(100.0, 100.0, 1.0, 0.05, 0.2, option_type="call", n_paths=40_000, rng=33)
    assert abs(res.estimate - black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)) < 4 * res.std_error + 0.02


def test_dividend_call_exercises_early():
    res = longstaff_schwartz(100.0, 100.0, 1.0, 0.05, 0.2, q=0.08, option_type="call", n_paths=40_000, rng=34)
    tree = CRRTree(1000).price(100.0, 100.0, 1.0, 0.05, 0.2, q=0.08, option_type="call", american=True).price
    assert abs(res.estimate - tree) < 4 * res.std_error + 0.03


def test_deep_in_the_money_exercises_at_once():
    res = longstaff_schwartz(10.0, 40.0, 1.0, 0.06, 0.2, n_paths=2_000, rng=35)
    assert res.estimate == 30.0 and res.std_error == 0.0


def test_regression_basis():
    X = regression_basis(np.array([0.5, 1.0, 2.0]), degree=3)
    assert X.shape == (3, 4)
    np.testing.assert_allclose(X[:, 1], np.exp(-0.5 * np.array([0.5, 1.0, 2.0])))
    np.testing.assert_allclose(regression_basis(np.array([2.0]), 2, "monomial"), [[1.0, 2.0, 4.0]])
    with pytest.raises(ValueError):
        regression_basis(np.array([1.0]), basis="chebyshev")


def test_longstaff_schwartz_without_antithetics_skips_dates_with_too_few_itm_paths():
    """A deep out-of-the-money put has almost no in-the-money paths to regress on, so it is never exercised."""
    res = longstaff_schwartz(200.0, 100.0, 0.5, 0.05, 0.2, n_steps=10, n_paths=2_000, antithetic=False, rng=36)
    assert res.n_samples == 2_000
    assert res.exercise_probability == 0.0
    assert res.estimate == pytest.approx(black_scholes_price(200.0, 100.0, 0.5, 0.05, 0.2, option_type="put"), abs=1e-5)
