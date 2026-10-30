"""Tests for the Acerbi-Tasche estimator, parametric ES, coherence and Euler allocation."""

import numpy as np
import pytest
from scipy import stats

from financialkit.constants import DEFAULT_RTOL
from financialkit.risk import (
    ExpectedShortfall,
    ValueAtRisk,
    check_coherence,
    euler_allocation,
    expected_shortfall,
    parametric_es,
    parametric_var,
    scenario_es_allocation,
)


def test_acerbi_tasche_matches_the_tail_mean_when_np_is_an_integer():
    x = np.random.default_rng(0).standard_normal(1000)
    assert expected_shortfall(x, 0.99) == pytest.approx(-np.sort(x)[:10].mean())


def test_acerbi_tasche_charges_the_fraction_of_the_quantile_atom():
    x = -np.arange(1.0, 101.0)  # worst losses 100, 99, 98, ...
    assert expected_shortfall(x, 0.975) == pytest.approx((100 + 99 + 0.5 * 98) / 2.5)


def test_sample_es_converges_to_the_normal_closed_form():
    rng = np.random.default_rng(1)
    n, alpha = 400_000, 0.975
    x = rng.standard_normal(n)
    est = expected_shortfall(x, alpha)
    tail = np.sort(x)[: int(n * (1 - alpha))]
    se = tail.std() / np.sqrt(tail.size)
    assert abs(est - parametric_es(0.0, 1.0, alpha)) < 4 * se


def test_t_es_matches_numerical_integration():
    nu, alpha = 5.0, 0.99
    scale = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(alpha, nu)
    integral = stats.t.expect(lambda x: x, args=(nu,), lb=q) / (1 - alpha)
    assert parametric_es(0.0, 1.0, alpha, distribution="t", df=nu) == pytest.approx(scale * integral, rel=1e-6)


def test_es_exceeds_var_and_scales_like_var():
    alpha = np.array([0.9, 0.99])
    assert np.all(parametric_es(0.0, 1.0, alpha) > parametric_var(0.0, 1.0, alpha))
    assert parametric_es(0.0, 2.0, 0.99, horizon=4) == pytest.approx(4 * parametric_es(0.0, 1.0, 0.99), rel=DEFAULT_RTOL)


def two_defaultable_bonds():
    """Artzner et al.'s counterexample: two independent bonds, each defaulting with probability 4%."""
    p = 0.04
    states = [(0, 0), (1, 0), (0, 1), (1, 1)]
    probs = [(1 - p) ** 2, p * (1 - p), p * (1 - p), p**2]
    counts = np.round(np.array(probs) * 10_000).astype(int)
    X = np.concatenate([np.full(c, -100.0 if a else 2.0) for (a, _), c in zip(states, counts, strict=True)])
    Y = np.concatenate([np.full(c, -100.0 if b else 2.0) for (_, b), c in zip(states, counts, strict=True)])
    return X, Y


def test_var_is_not_subadditive_but_es_is():
    X, Y = two_defaultable_bonds()
    var = check_coherence(ValueAtRisk(0.95), X, Y)
    es = check_coherence(ExpectedShortfall(0.95), X, Y)
    assert var.rho_x < 0 and var.rho_sum > 0 and not var.subadditivity and not var.coherent
    assert var.monotonicity and var.positive_homogeneity and var.translation_invariance
    assert es.coherent


def test_risk_measure_validates_alpha():
    with pytest.raises(ValueError):
        ExpectedShortfall(1.5)
    assert repr(ValueAtRisk(0.99)) == "ValueAtRisk(alpha=0.99)"


@pytest.mark.parametrize("measure", ["var", "es", "volatility"])
def test_euler_contributions_sum_and_equal_marginal_times_weight(measure):
    rng = np.random.default_rng(2)
    A = rng.standard_normal((4, 4))
    cov = A @ A.T / 4 * 1e-4
    w = np.array([3.0, -1.0, 2.0, 0.5])
    res = euler_allocation(w, cov, 0.99, measure)
    assert res.contributions.sum() == pytest.approx(res.total, rel=DEFAULT_RTOL)
    h = 1e-6
    for i in range(4):
        e = np.zeros(4)
        e[i] = h
        fd = (euler_allocation(w + e, cov, 0.99, measure).total - euler_allocation(w - e, cov, 0.99, measure).total) / (2 * h)
        assert w[i] * fd == pytest.approx(res.contributions[i], rel=1e-6)


def test_scenario_es_allocation_adds_up_to_portfolio_es():
    rng = np.random.default_rng(3)
    X = rng.multivariate_normal([0, 0, 0], [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 4]], size=20_000)
    res = scenario_es_allocation(X, 0.99)
    assert res.total == pytest.approx(expected_shortfall(X.sum(axis=1), 0.99))
    assert res.shares.sum() == pytest.approx(1.0)
    assert np.argmax(res.contributions) == 2


def test_risk_inputs_are_validated():
    with pytest.raises(ValueError, match="alpha must lie"):
        parametric_es(0.0, 1.0, alpha=1.0)
    with pytest.raises(ValueError, match="df > 2"):
        parametric_es(0.0, 1.0, distribution="t", df=2.0)
    with pytest.raises(ValueError, match="distribution must be"):
        parametric_es(0.0, 1.0, distribution="cauchy")
    with pytest.raises(ValueError, match="measure must be"):
        euler_allocation([0.5, 0.5], np.eye(2), measure="sharpe")
    with pytest.raises(ValueError, match="n_scenarios, n_positions"):
        scenario_es_allocation(np.zeros(10))
    with pytest.raises(ValueError, match="same states"):
        check_coherence(ValueAtRisk(0.95), np.zeros(3), np.zeros(4))
