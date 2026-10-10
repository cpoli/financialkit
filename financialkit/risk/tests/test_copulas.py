"""Tests for Gaussian and t copulas, Sklar's construction and tail dependence."""

import numpy as np
import pytest
from scipy import stats

from financialkit.risk import (
    apply_marginals,
    copula_kendall_tau,
    empirical_tail_dependence,
    gaussian_copula_sample,
    pseudo_observations,
    t_copula_sample,
    t_tail_dependence,
)

R = np.array([[1.0, 0.5], [0.5, 1.0]])


@pytest.mark.parametrize("sampler", [lambda n, rng: gaussian_copula_sample(R, n, rng), lambda n, rng: t_copula_sample(R, 4, n, rng)])
def test_copula_marginals_are_uniform_and_tau_matches(sampler):
    n = 4000
    u = sampler(n, 0)
    for j in range(2):
        assert stats.kstest(u[:, j], "uniform").pvalue > 1e-3
    tau = stats.kendalltau(u[:, 0], u[:, 1]).statistic
    # sd of Kendall's tau is below sqrt(4 / (9 n)) * 1.0 for |tau| <= 0.5.
    assert abs(tau - copula_kendall_tau(0.5)) < 4 * np.sqrt(4 / (9 * n))


def test_t_copula_with_infinite_df_is_gaussian():
    np.testing.assert_allclose(t_copula_sample(R, np.inf, 100, 1), gaussian_copula_sample(R, 100, 1))


def test_sklar_construction_keeps_marginals_and_ranks():
    u = gaussian_copula_sample(R, 2000, 2)
    x = apply_marginals(u, [stats.t(3).ppf, stats.expon.ppf])
    assert stats.kstest(x[:, 1], "expon").pvalue > 1e-3
    np.testing.assert_allclose(pseudo_observations(x), pseudo_observations(u))
    with pytest.raises(ValueError):
        apply_marginals(u, [stats.norm.ppf])


def test_t_tail_dependence_closed_form():
    """Demarta and McNeil (2005), Table 1."""
    assert t_tail_dependence(0.5, 4) == pytest.approx(0.25, abs=0.005)
    assert t_tail_dependence(0.0, 2) == pytest.approx(0.18, abs=0.005)
    np.testing.assert_allclose(t_tail_dependence(0.5, [np.inf, 1e6]), [0.0, 0.0], atol=1e-6)
    assert t_tail_dependence(1.0, np.inf) == 1.0


def test_empirical_tail_dependence_matches_closed_form_far_out():
    n, q = 400_000, 0.002
    u = t_copula_sample(R, 4, n, 3)
    est = empirical_tail_dependence(u[:, 0], u[:, 1], q)
    upper = empirical_tail_dependence(u[:, 0], u[:, 1], q, tail="upper")
    lam = t_tail_dependence(0.5, 4)
    se = np.sqrt(lam * (1 - lam) / (n * q))
    # Finite-q bias is positive and of order q^(1/nu); allow 4 se plus a 0.05 bias band.
    assert abs(est - lam) < 4 * se + 0.05 and abs(upper - lam) < 4 * se + 0.05
    g = gaussian_copula_sample(R, n, 4)
    assert empirical_tail_dependence(g[:, 0], g[:, 1], q) < est - 4 * se
