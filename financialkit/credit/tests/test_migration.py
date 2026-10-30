"""Tests for CreditMetrics migration, generator matrices and their regularization."""

import numpy as np
import pytest

from financialkit.credit import (
    RATINGS,
    check_transition_matrix,
    creditmetrics_bbb_values,
    creditmetrics_bond,
    creditmetrics_matrix,
    creditmetrics_portfolio,
    generator_from_transition,
    is_valid_generator,
    jlt_risky_zero,
    joint_migration_probabilities,
    migration_thresholds,
    regularize_generator,
    simulate_migrations,
    transition_from_generator,
)

P = creditmetrics_matrix()


def test_creditmetrics_bbb_bond():
    """CreditMetrics Technical Document (1997), Chapter 3: mean 107.09, std 2.99."""
    res = creditmetrics_bond(creditmetrics_matrix(normalize=False)[3], creditmetrics_bbb_values())
    assert res.mean == pytest.approx(107.09, abs=0.005)
    assert res.std == pytest.approx(2.99, abs=0.005)
    # 1% percentile: default (0.18%) plus CCC (0.12%) plus B (1.17%) first exceeds 1% at B.
    assert res.percentile(0.01) == pytest.approx(98.10)
    assert res.credit_var(0.99) == pytest.approx(res.mean - 98.10)


def test_thresholds_reproduce_row():
    from scipy import stats

    t = migration_thresholds(P[3])
    assert t[0] == np.inf and t[-1] == -np.inf and np.all(np.diff(t) < 0)
    np.testing.assert_allclose(stats.norm.cdf(t[:-1]) - stats.norm.cdf(t[1:]), P[3], atol=1e-12)


def test_joint_migrations_marginals_and_independence():
    J = joint_migration_probabilities(P[3], P[2], 0.3)
    np.testing.assert_allclose(J.sum(axis=1), P[3], atol=1e-6)
    np.testing.assert_allclose(J.sum(axis=0), P[2], atol=1e-6)
    J0 = joint_migration_probabilities(P[3], P[2], 0.0)
    np.testing.assert_allclose(J0, np.outer(P[3], P[2]), atol=1e-6)
    # Positive correlation raises the joint default probability.
    assert J[-1, -1] > J0[-1, -1]


def test_simulated_migrations_match_matrix_row():
    n = 40000
    new = simulate_migrations(P, [4], 0.2, n, rng=0)[:, 0]
    freq = np.bincount(new, minlength=8) / n
    se = np.sqrt(P[4] * (1 - P[4]) / n)
    assert np.all(np.abs(freq - P[4]) < 4 * se + 1e-12)


def test_portfolio_mean_is_sum_of_bond_means():
    vals = creditmetrics_bbb_values()
    ratings = [3, 3, 2, 4]
    res = creditmetrics_portfolio(P, ratings, vals, 0.25, 20000, rng=1)
    exact = sum(P[g] @ vals for g in ratings)
    assert abs(res.mean - exact) < 4 * res.std / np.sqrt(20000)


def test_generator_round_trip_and_regularization():
    """Israel, Rosenthal and Wei (2001): the log of the CreditMetrics matrix has
    negative off-diagonal entries; both adjustments give valid generators close to it."""
    Q = generator_from_transition(P)
    np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(transition_from_generator(Q), P, atol=1e-10)
    assert not is_valid_generator(Q)
    for method in ("diagonal", "weighted"):
        G = regularize_generator(Q, method)
        assert is_valid_generator(G)
        Pt = transition_from_generator(G)
        np.testing.assert_allclose(Pt.sum(axis=1), 1.0, atol=1e-12)
        assert np.abs(Pt - P).max() < 2e-3
    with pytest.raises(ValueError):
        regularize_generator(Q, "bad")


def test_generator_semigroup_and_jlt():
    G = regularize_generator(generator_from_transition(P))
    P2 = transition_from_generator(G, [1.0, 2.0])
    np.testing.assert_allclose(P2[1], P2[0] @ P2[0], atol=1e-12)
    # JLT bond: no default risk from AAA over a tiny horizon, and spreads widen down the scale.
    T = 5.0
    prices = np.array([jlt_risky_zero(T, 0.03, G, i, 0.4) for i in range(7)])
    assert np.all(np.diff(prices) < 0)
    pd = transition_from_generator(G, T)[3, -1]
    assert prices[3] == pytest.approx(np.exp(-0.15) * (0.4 + 0.6 * (1 - pd)))
    doubled = jlt_risky_zero(T, 0.03, G, 3, 0.4, premia=np.full(8, 2.0))
    assert doubled < prices[3]
    assert len(RATINGS) == 8


def test_migration_inputs_are_validated():
    with pytest.raises(ValueError, match="same shape"):
        creditmetrics_bond([0.5, 0.5], [100.0, 90.0, 0.0])
    with pytest.raises(ValueError, match="positive diagonal"):
        generator_from_transition([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="square"):
        check_transition_matrix(np.ones((2, 3)) / 3)
    with pytest.raises(ValueError, match="rows summing to one"):
        check_transition_matrix([[0.9, 0.2], [0.0, 1.0]])
