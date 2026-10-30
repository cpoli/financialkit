"""Tests for Hansen's GMM and the Newey-West covariance."""

import numpy as np
import pytest

from financialkit.asset_pricing import gmm, newey_west
from financialkit.constants import DEFAULT_ATOL


def test_gmm_mean_is_the_sample_mean():
    x = np.random.default_rng(0).normal(1.0, 2.0, 500)
    res = gmm(lambda th: (x - th[0])[:, None], [0.0], weighting="identity")
    assert res.params[0] == pytest.approx(x.mean())
    assert res.std_errors[0] == pytest.approx(x.std() / np.sqrt(x.size), rel=1e-6)
    assert res.df == 0 and np.isnan(res.j_pvalue)


def test_gmm_ols_gives_white_standard_errors():
    """Moments x (y - x'b): OLS estimates with White's heteroskedasticity-robust covariance."""
    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(400), rng.standard_normal(400)])
    y = x @ [0.5, 2.0] + (1 + np.abs(x[:, 1])) * rng.standard_normal(400)
    res = gmm(lambda b: x * (y - x @ b)[:, None], [0.0, 0.0])
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    e = y - x @ b
    Q = np.linalg.inv(x.T @ x)
    white = Q @ (x.T * e**2) @ x @ Q
    np.testing.assert_allclose(res.params, b, rtol=1e-7)
    np.testing.assert_allclose(res.cov, white, rtol=1e-5)
    assert res.t_values == pytest.approx(res.params / res.std_errors)


def _iv_moments(seed, n=2000):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    u = rng.standard_normal(n)
    x = z @ [1.0, 0.5, -0.5] + 0.5 * u + rng.standard_normal(n)
    y = 2.0 * x + u
    return lambda th: z * (y - th[0] * x)[:, None]


def test_overidentified_iv_recovers_the_slope_and_passes_j():
    res = gmm(_iv_moments(2), [0.0])
    assert abs(res.params[0] - 2.0) < 4 * res.std_errors[0]
    assert res.df == 2 and res.j_pvalue > 0.01


def test_identity_j_matches_efficient_j_when_weighting_is_efficient():
    """Hansen's generalised-inverse J equals T g'S^{-1}g at the efficient weighting."""
    moments = _iv_moments(3)
    eff = gmm(moments, [0.0], weighting="iterated")
    fixed = gmm(moments, [0.0], weighting=eff.weighting)
    assert fixed.params[0] == pytest.approx(eff.params[0], rel=1e-8)
    assert fixed.j_stat == pytest.approx(eff.j_stat, rel=1e-4)


def test_iterated_gmm_is_a_fixed_point():
    moments = _iv_moments(4)
    it = gmm(moments, [0.0], weighting="iterated")
    again = gmm(moments, it.params, weighting=it.weighting)
    assert again.params[0] == pytest.approx(it.params[0], abs=1e-8)


def test_gmm_rejects_bad_inputs():
    m = _iv_moments(5, 50)
    with pytest.raises(ValueError, match="at least as many moments"):
        gmm(lambda th: m(th)[:, :1], [0.0, 1.0])
    with pytest.raises(ValueError, match="weighting must be"):
        gmm(m, [0.0], weighting="optimal")
    with pytest.raises(ValueError, match="shape"):
        gmm(m, [0.0], weighting=np.eye(2))


def test_newey_west_white_and_long_run_variance():
    rng = np.random.default_rng(6)
    u = rng.standard_normal((300, 2))
    np.testing.assert_allclose(newey_west(u), u.T @ u / 300, atol=DEFAULT_ATOL)
    # MA(1) e_t + e_{t-1}: long-run variance (1 + 1)^2 = 4, against a variance of 2.
    e = rng.standard_normal(200_001)
    ma = e[1:] + e[:-1]
    assert newey_west(ma, lags=40)[0, 0] == pytest.approx(4.0, rel=0.05)
    with pytest.raises(ValueError, match="nonnegative"):
        newey_west(ma, lags=-1)
