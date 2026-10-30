"""Tests for the Andersen-Sidenius-Basu loss recursion."""

import itertools

import numpy as np
import pytest
from scipy import stats

from financialkit.constants import DEFAULT_ATOL
from financialkit.credit import (
    asb_loss_distribution,
    conditional_default_probability,
    expected_tranche_loss,
    loss_recursion,
    normal_factor_quadrature,
    one_factor_loss_distribution,
)


def test_recursion_is_the_binomial_law_for_alike_names():
    f = loss_recursion(np.full(30, 0.07), np.ones(30, dtype=int))
    np.testing.assert_allclose(f, stats.binom.pmf(np.arange(31), 30, 0.07), atol=DEFAULT_ATOL)


def test_recursion_matches_enumeration_with_unequal_losses():
    """All 2^5 default patterns of five names with losses 1, 2, 2, 3, 5 units."""
    p = np.array([0.1, 0.3, 0.05, 0.2, 0.15])
    u = np.array([1, 2, 2, 3, 5])
    expected = np.zeros(u.sum() + 1)
    for pattern in itertools.product([0, 1], repeat=5):
        d = np.array(pattern)
        expected[d @ u] += np.prod(np.where(d == 1, p, 1 - p))
    np.testing.assert_allclose(loss_recursion(p, u), expected, atol=DEFAULT_ATOL)


def test_recursion_rows_are_scenarios():
    P = np.array([[0.1, 0.2], [0.5, 0.9]])
    f = loss_recursion(P, [1, 1])
    for row, p in zip(f, P, strict=True):
        np.testing.assert_allclose(row, loss_recursion(p, [1, 1]), atol=DEFAULT_ATOL)


def test_asb_matches_the_homogeneous_binomial_mixture():
    asb = asb_loss_distribution(np.full(100, 0.03), 0.2, unit=0.6 / 100, n_nodes=96)
    ref = one_factor_loss_distribution(0.03, 0.2, 100, lgd=0.6, n_nodes=96)
    np.testing.assert_allclose(asb.losses, ref.losses, atol=DEFAULT_ATOL)
    np.testing.assert_allclose(asb.pmf, ref.pmf, atol=DEFAULT_ATOL)


def test_asb_heterogeneous_matches_enumeration_on_the_factor_grid():
    """Four names with their own p, rho and losses, enumerated given each factor node."""
    p = np.array([0.02, 0.05, 0.1, 0.01])
    rho = np.array([0.1, 0.3, 0.5, 0.2])
    u = np.array([3, 1, 2, 4])
    asb = asb_loss_distribution(p, rho, u, unit=0.25, n_nodes=40)
    x, w = normal_factor_quadrature(40)
    expected = np.zeros(u.sum() + 1)
    for xi, wi in zip(x, w, strict=True):
        pm = np.asarray(conditional_default_probability(p, rho, xi))
        for pattern in itertools.product([0, 1], repeat=4):
            d = np.array(pattern)
            expected[d @ u] += wi * np.prod(np.where(d == 1, pm, 1 - pm))
    np.testing.assert_allclose(asb.pmf, expected, atol=DEFAULT_ATOL)
    np.testing.assert_allclose(asb.losses, 0.25 * np.arange(u.sum() + 1))
    # the expected loss does not depend on the correlations
    assert np.isclose(asb.mean, 0.25 * p @ u, atol=1e-10)


def test_asb_against_monte_carlo():
    """Simulated one-factor defaults of a mixed pool fall within 4 standard errors."""
    rng = np.random.default_rng(7)
    n = 40
    p = rng.uniform(0.005, 0.05, n)
    u = rng.integers(1, 4, n)
    asb = asb_loss_distribution(p, 0.3, u)
    sims = 200_000
    M = rng.standard_normal((sims, 1))
    Z = np.sqrt(0.3) * M + np.sqrt(0.7) * rng.standard_normal((sims, n))
    losses = (Z < stats.norm.ppf(p)) @ u
    tail = asb.losses >= 10
    prob_mc = np.mean(losses >= 10)
    se = np.sqrt(prob_mc * (1 - prob_mc) / sims)
    assert abs(prob_mc - asb.pmf[tail].sum()) < 4 * se


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p": [0.1, 1.2], "rho": 0.2},
        {"p": [0.1, 0.2], "rho": 1.0},
        {"p": [[0.1, 0.2]], "rho": 0.2},
        {"p": [0.1, 0.2], "rho": 0.2, "loss_units": [1, 1.5]},
        {"p": [0.1, 0.2], "rho": 0.2, "loss_units": [1, 2, 3]},
    ],
)
def test_asb_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        asb_loss_distribution(**kwargs)


def test_binomial_mixtures_survive_extreme_conditional_probabilities():
    """At rho = 0.99 one Gauss-Hermite node gives p(M) = 9e-308, where scipy 1.18's binomial overflowed."""
    p = 0.04686621292249526
    assert np.isclose(one_factor_loss_distribution(p, 0.99, 125, n_nodes=96).pmf.sum(), 1.0)
    assert 0.0 < expected_tranche_loss(0.0, 0.03, p, 0.99, n_names=125) < 1.0
