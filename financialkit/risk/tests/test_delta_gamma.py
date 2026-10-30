"""Tests for Rouvinez's delta-gamma VaR and the BDSS liquidity-adjusted VaR."""

import numpy as np
import pytest
from scipy import stats

from financialkit.constants import DEFAULT_RTOL
from financialkit.risk import (
    delta_gamma_cdf,
    delta_gamma_cumulants,
    delta_gamma_var,
    delta_normal_var,
    exogenous_liquidity_cost,
    liquidity_adjusted_var,
)

RNG = np.random.default_rng(0)
A = RNG.standard_normal((3, 3))
COV = A @ A.T / 3
G = RNG.standard_normal((3, 3))
GAMMA = G + G.T
DELTA = 0.5 * RNG.standard_normal(3)


def sample_quadratic(n, theta=0.0, seed=1):
    X = np.random.default_rng(seed).multivariate_normal(np.zeros(3), COV, n)
    return theta + X @ DELTA + 0.5 * np.einsum("ni,ij,nj->n", X, GAMMA, X)


@pytest.mark.parametrize("alpha", [0.95, 0.99, 0.999])
def test_one_factor_gamma_is_a_scaled_chi_squared(alpha):
    """Short gamma: the loss is -1/2 g s^2 chi^2_1, so the VaR is that times the chi-squared alpha quantile."""
    g, s2 = -4.0, 0.3
    expected = -0.5 * g * s2 * stats.chi2.ppf(alpha, 1)
    assert delta_gamma_var([0.0], [[g]], [[s2]], alpha) == pytest.approx(expected, rel=1e-7)
    # Long gamma: the 1 - alpha quantile of a positive variable, so a negative VaR.
    assert delta_gamma_var([0.0], [[-g]], [[s2]], alpha) == pytest.approx(0.5 * g * s2 * stats.chi2.ppf(1 - alpha, 1), rel=1e-6)


def test_cdf_at_the_edge_of_a_one_sided_support():
    """Long gamma alone is 1/2 chi^2_1 >= 0: the cdf is 0 at 0 and chi^2_1(2x) just above it."""
    assert delta_gamma_cdf(0.0, [0.0], [[1.0]], [[1.0]]) == pytest.approx(0.0, abs=1e-6)
    assert delta_gamma_cdf(1e-12, [0.0], [[-1.0]], [[1.0]]) == pytest.approx(1.0, abs=1e-5)
    assert delta_gamma_cdf(0.5, [0.0], [[1.0]], [[1.0]]) == pytest.approx(stats.chi2.cdf(1.0, 1), abs=1e-8)


def test_no_gamma_is_delta_normal():
    for method in ("exact", "cornish_fisher"):
        got = delta_gamma_var(DELTA, np.zeros((3, 3)), COV, 0.99, horizon=10, method=method)
        assert got == pytest.approx(delta_normal_var(DELTA, COV, 0.99, horizon=10), rel=1e-8)


def test_cumulants_match_simulated_moments():
    q = sample_quadratic(400_000, theta=0.1)
    k = delta_gamma_cumulants(DELTA, GAMMA, COV, theta=0.1)
    assert abs(q.mean() - k[0]) < 4 * q.std() / np.sqrt(q.size)
    assert q.var() == pytest.approx(k[1], rel=0.02)  # sampling error of a variance with kurtosis ~ 10: about 0.5%
    assert stats.skew(q) * q.std() ** 3 == pytest.approx(k[2], rel=0.05)


def test_exact_cdf_matches_simulation():
    q = sample_quadratic(400_000, theta=0.1)
    x = np.array([-5.0, -2.0, 0.0, 1.0, 3.0])
    p_mc = (q[:, None] <= x).mean(axis=0)
    se = np.sqrt(p_mc * (1 - p_mc) / q.size) + 1e-6
    assert np.all(np.abs(delta_gamma_cdf(x, DELTA, GAMMA, COV, theta=0.1) - p_mc) < 4 * se)


def test_exact_var_inverts_the_cdf_and_beats_cornish_fisher():
    v = delta_gamma_var(DELTA, GAMMA, COV, 0.99)
    assert delta_gamma_cdf(-v, DELTA, GAMMA, COV) == pytest.approx(0.01, abs=1e-9)
    q = sample_quadratic(400_000)
    mc = -np.quantile(q, 0.01)
    assert abs(v - mc) < abs(delta_gamma_var(DELTA, GAMMA, COV, 0.99, method="cornish_fisher") - mc)


def test_horizon_scales_the_covariance():
    assert delta_gamma_var(DELTA, GAMMA, COV, horizon=2.0) == pytest.approx(delta_gamma_var(DELTA, GAMMA, 2 * COV), rel=DEFAULT_RTOL)


def test_degenerate_and_invalid_inputs():
    assert delta_gamma_var([0.0, 0.0], np.zeros((2, 2)), np.eye(2), theta=-0.5) == pytest.approx(0.5)
    assert delta_gamma_cdf(0.0, [0.0], [[0.0]], [[1.0]], theta=-1.0) == 1.0
    with pytest.raises(ValueError, match="agree"):
        delta_gamma_var([1.0, 2.0], [[1.0]], np.eye(2))
    with pytest.raises(ValueError, match="method"):
        delta_gamma_var([1.0], [[1.0]], [[1.0]], method="delta")


def test_liquidity_cost_is_half_the_stressed_spread():
    assert exogenous_liquidity_cost(-2e6, 0.004, 0.001, 3.0) == pytest.approx(0.5 * 2e6 * 0.007)
    with pytest.raises(ValueError, match="non-negative"):
        exogenous_liquidity_cost(1.0, -0.01)


def test_liquidity_adjusted_var_adds_the_cost_to_a_lognormal_var():
    """Bangia et al. (1999), eq. (4): LVaR = P(1 - exp(mu - z sigma)) + P (S + a sigma_S) / 2."""
    z = stats.norm.ppf(0.99)
    market = 1e6 * (1 - np.exp(-z * 0.02 * np.sqrt(10)))
    assert liquidity_adjusted_var(1e6, 0.02, 0.0, horizon=10) == pytest.approx(market, rel=DEFAULT_RTOL)
    got = liquidity_adjusted_var(1e6, 0.02, 0.005, 0.002, horizon=10)
    assert got == pytest.approx(market + 0.5e6 * (0.005 + z * 0.002), rel=DEFAULT_RTOL)
    assert liquidity_adjusted_var(1e6, 0.02, 0.005, 0.002, horizon=10, a=4.5) > got


def test_liquidity_adjusted_var_broadcasts():
    out = liquidity_adjusted_var(1e6, 0.02, [0.0, 0.01, 0.02], 0.005)
    assert out.shape == (3,) and np.all(np.diff(out) == pytest.approx(0.5e6 * 0.01))
    with pytest.raises(ValueError, match="alpha"):
        liquidity_adjusted_var(1e6, 0.02, 0.01, alpha=1.0)
