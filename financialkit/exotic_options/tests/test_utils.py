"""Tests for financialkit.exotic_options.utils.numerics."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import kstest, norm

from financialkit.exotic_options import barrier_crossing_probability, bivariate_normal_cdf, bridge_extremum, parse_barrier_type


def _bvn_quad(h, k, rho):
    return quad(lambda x: norm.pdf(x) * norm.cdf((k - rho * x) / np.sqrt(1 - rho**2)), -np.inf, h, epsabs=1e-14, epsrel=1e-12)[0]


def test_bivariate_normal_against_quadrature():
    rng = np.random.default_rng(0)
    pts = [(0.0, 0.0, 0.5), (0.0, 1.0, 0.3), (1.0, 0.0, -0.4), (0.0, -1.0, 0.7), (2.0, -3.0, 0.95), (-8.0, -8.0, 0.5)]
    pts += [(*rng.normal(size=2) * 2.5, rng.uniform(-0.99, 0.99)) for _ in range(60)]
    h, k, rho = np.array(pts).T
    np.testing.assert_allclose(bivariate_normal_cdf(h, k, rho), [_bvn_quad(*p) for p in pts], atol=1e-12)
    assert bivariate_normal_cdf(1.0, 2.0, 0.0) == pytest.approx(norm.cdf(1.0) * norm.cdf(2.0), abs=1e-14)
    with pytest.raises(ValueError):
        bivariate_normal_cdf(0.0, 0.0, 1.0)


def test_parse_barrier_type():
    assert parse_barrier_type("down-and-out") == (True, True)
    assert parse_barrier_type("up-and-in") == (False, False)
    with pytest.raises(ValueError):
        parse_barrier_type("knock-out")


def test_crossing_probability_matches_reflection_principle():
    """For a bridge from 0 to 0 over unit time the maximum exceeds b with probability exp(-2 b^2)."""
    assert float(barrier_crossing_probability(0.0, 0.0, 0.7, 1.0, 1.0)) == pytest.approx(np.exp(-2 * 0.49))
    np.testing.assert_array_equal(barrier_crossing_probability([0.0, 1.0], [2.0, 2.0], 1.0, 1.0, 1.0), [1.0, 1.0])


def test_bridge_extremum_is_the_brownian_maximum():
    """Driftless BM on [0, 1] from 0: P(max <= m) = 2 N(m) - 1, sampled from a single step."""
    rng = np.random.default_rng(1)
    end = rng.standard_normal(20_000)
    paths = np.column_stack([np.zeros_like(end), end])
    m = bridge_extremum(paths, 1.0, 1.0, "max", rng=rng)
    assert kstest(m, lambda x: np.clip(2 * norm.cdf(x) - 1, 0, 1)).pvalue > 1e-3
    lo = bridge_extremum(paths, 1.0, 1.0, "min", rng=2)
    assert np.all(lo <= np.minimum(0.0, end)) and np.all(m >= np.maximum(0.0, end))
    with pytest.raises(ValueError):
        bridge_extremum(paths, 1.0, 1.0, "median")
