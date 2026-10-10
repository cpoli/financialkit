"""Tests for financialkit.monte_carlo.systems.variance_reduction."""

import numpy as np
import pytest
from scipy.stats import norm

from financialkit.monte_carlo import (
    antithetic_estimate,
    control_variate_estimate,
    importance_sampling_normal,
    latin_hypercube_estimate,
    moment_match,
    moment_matching_estimate,
    neyman_allocation,
    stratified_estimate,
    terminal_prices,
    vanilla_payoff,
)
from financialkit.options import black_scholes_price

S, K, T, r, SIG = 100.0, 100.0, 1.0, 0.05, 0.2
EXACT = float(black_scholes_price(S, K, T, r, SIG))


def call_of_z(z):
    return np.exp(-r * T) * vanilla_payoff(terminal_prices(S, T, r, SIG, z), K)


def call_of_u(u):
    return call_of_z(norm.ppf(u))


def crude_variance():
    return call_of_z(np.random.default_rng(99).standard_normal(200_000)).var()


def test_neyman_allocation_proportional_to_weight_times_std():
    alloc = neyman_allocation([0.25, 0.25, 0.5], [2.0, 2.0, 1.0], 1000)
    assert alloc.sum() == 1000
    assert np.all(np.abs(alloc - 1000 / 3) < 1)
    assert neyman_allocation([0.5, 0.5], [0.0, 1.0], 10).min() == 2
    with pytest.raises(ValueError):
        neyman_allocation([0.5, 0.5], [1.0, 1.0], 3)


@pytest.mark.parametrize("allocation", ["proportional", "neyman"])
def test_stratified_unbiased_and_lower_variance(allocation):
    res = stratified_estimate(call_of_u, 20_000, n_strata=20, allocation=allocation, rng=1)
    assert abs(res.estimate - EXACT) < 4 * res.std_error
    assert res.variance < 0.1 * crude_variance()
    assert res.allocation.sum() == 20_000


def test_neyman_beats_proportional():
    prop = stratified_estimate(call_of_u, 20_000, n_strata=10, allocation="proportional", rng=2)
    ney = stratified_estimate(call_of_u, 20_000, n_strata=10, allocation="neyman", rng=2)
    assert ney.std_error < prop.std_error
    with pytest.raises(ValueError):
        stratified_estimate(call_of_u, 100, allocation="optimal")


def test_importance_sampling_rare_event():
    exact = norm.sf(5.0)
    res = importance_sampling_normal(lambda z: (z > 5.0).astype(float), 20_000, shift=5.0, rng=3)
    assert abs(res.estimate - exact) < 4 * res.std_error
    assert res.std_error / exact < 0.02


def test_importance_sampling_multivariate_shift():
    theta = np.array([1.0, 2.0])
    res = importance_sampling_normal(lambda z: z.sum(axis=1), 100_000, shift=theta, rng=4)
    assert abs(res.estimate) < 4 * res.std_error


def test_antithetic_reduces_variance_for_monotone_payoff():
    res = antithetic_estimate(call_of_z, 50_000, rng=5)
    assert res.n_samples == 100_000
    assert abs(res.estimate - EXACT) < 4 * res.std_error
    assert res.variance < 0.7 * crude_variance()


def test_antithetic_vector_dimension():
    res = antithetic_estimate(lambda z: np.exp(z.sum(axis=1)), 50_000, dim=2, rng=6)
    assert abs(res.estimate - np.e) < 4 * res.std_error


def test_latin_hypercube_additive_integrand():
    # Additive integrand: LHS removes almost all of the variance.
    f = lambda u: np.sum(u**2, axis=1)  # noqa: E731
    res = latin_hypercube_estimate(f, 1000, dim=4, n_replications=20, rng=7)
    assert abs(res.estimate - 4 / 3) < 4 * res.std_error
    assert res.variance < 0.01 * (4 * 4 / 45)


def test_control_variate_on_terminal_price():
    z = np.random.default_rng(8).standard_normal(50_000)
    S_T = terminal_prices(S, T, r, SIG, z)
    res = control_variate_estimate(call_of_z(z), np.exp(-r * T) * S_T, S)
    assert abs(res.estimate - EXACT) < 4 * res.std_error
    assert 0.8 < res.r_squared < 1.0
    assert res.std_error == pytest.approx(res.crude_std_error * np.sqrt(1 - res.r_squared), rel=0.01)


def test_control_variate_two_controls_exact_fit():
    x = np.random.default_rng(9).standard_normal((1000, 2))
    res = control_variate_estimate(1.0 + 2 * x[:, 0] - x[:, 1], x, [0.0, 0.0])
    assert res.estimate == pytest.approx(1.0)
    np.testing.assert_allclose(res.beta, [2.0, -1.0])


def test_moment_matching():
    w = moment_match(np.random.default_rng(10).standard_normal(100))
    assert w.mean() == pytest.approx(0.0, abs=1e-12)
    assert w.std() == pytest.approx(1.0)
    res = moment_matching_estimate(call_of_z, 5000, n_replications=20, rng=11)
    assert abs(res.estimate - EXACT) < 4 * res.std_error
    assert res.variance < 0.5 * crude_variance()
