"""Tests for Vasicek's LHP, CreditRisk+, Li's copula, the Basel IRB formula and tranches."""

import numpy as np
import pytest
from scipy import integrate, stats

from financialkit.credit import (
    HazardCurve,
    basel_correlation,
    basel_irb_capital,
    basel_maturity_adjustment,
    basel_risk_weight,
    copula_default_times,
    creditrisk_plus,
    default_correlation,
    expected_tranche_loss,
    joint_default_probability,
    nth_to_default_probability,
    one_factor_loss_distribution,
    panjer_recursion,
    simulate_one_factor_defaults,
    tranche_loss,
    tranche_spread,
    vasicek_cdf,
    vasicek_pdf,
    vasicek_quantile,
)


def test_vasicek_distribution_is_consistent():
    p, rho = 0.02, 0.15
    assert integrate.quad(lambda x: vasicek_pdf(x, p, rho), 0, 1, limit=200)[0] == pytest.approx(1.0, abs=1e-6)
    assert integrate.quad(lambda x: x * vasicek_pdf(x, p, rho), 0, 1, limit=200)[0] == pytest.approx(p, abs=1e-7)
    q = np.array([0.5, 0.9, 0.999])
    np.testing.assert_allclose(vasicek_cdf(vasicek_quantile(q, p, rho), p, rho), q, rtol=1e-12)


def test_large_pool_default_fraction_converges_to_vasicek():
    p, rho, n_obl, n = 0.03, 0.2, 5000, 20000
    frac = simulate_one_factor_defaults(p, rho, n_obl, n, rng=0) / n_obl
    for x in (0.01, 0.03, 0.08):
        est = np.mean(frac <= x)
        # Finite-pool binomial noise shifts the CDF by O(1/sqrt(n_obl)); allow 0.01 beyond 4 se.
        assert abs(est - vasicek_cdf(x, p, rho)) < 4 * np.sqrt(est * (1 - est) / n) + 0.01


def test_one_factor_quadrature_matches_monte_carlo():
    p, rho, m = 0.05, 0.3, 50
    dist = one_factor_loss_distribution(p, rho, m)
    assert dist.pmf.sum() == pytest.approx(1.0, abs=1e-10)
    assert dist.mean == pytest.approx(p, rel=1e-8)
    k = simulate_one_factor_defaults(p, rho, m, 40000, rng=1)
    for j in (0, 2, 10):
        est = np.mean(k == j)
        assert abs(est - dist.pmf[j]) < 4 * np.sqrt(dist.pmf[j] * (1 - dist.pmf[j]) / k.size)


def test_panjer_negative_binomial():
    a, alpha = 0.3, 2.5
    f = panjer_recursion([0.0, 1.0], a, (alpha - 1) * a, (1 - a) ** alpha, 30)
    np.testing.assert_allclose(f, stats.nbinom.pmf(np.arange(31), alpha, 1 - a), rtol=1e-12)


def test_creditrisk_plus_closed_forms():
    """CSFB (1997), Section A: unit exposures give a Poisson count (sigma = 0)
    and a negative binomial count with gamma-mixed rates."""
    p = np.full(50, 0.02)
    mu = p.sum()
    pois = creditrisk_plus(np.ones(50), p, sigma=0.0, n_max=20)
    np.testing.assert_allclose(pois.pmf, stats.poisson.pmf(np.arange(21), mu), rtol=1e-12)
    sigma = 0.8
    nb = creditrisk_plus(np.ones(50), p, sigma=sigma, n_max=30)
    alpha, delta = 1 / sigma**2, mu * sigma**2 / (1 + mu * sigma**2)
    np.testing.assert_allclose(nb.pmf, stats.nbinom.pmf(np.arange(31), alpha, 1 - delta), rtol=1e-10)
    # Mean and variance with mixed exposures (CSFB eqs. for one sector).
    E, pd = np.array([1.0, 2.0, 5.0, 5.0]), np.array([0.03, 0.02, 0.01, 0.04])
    d = creditrisk_plus(E, pd, sigma=0.6, n_max=400)
    assert d.mean == pytest.approx(pd @ E, rel=1e-9)
    assert d.std**2 == pytest.approx(pd @ E**2 + 0.36 * (pd @ E) ** 2, rel=1e-8)
    assert d.unexpected_loss(0.99) > 0
    with pytest.raises(ValueError):
        creditrisk_plus([1.0], [0.1], sigma=-1)


def test_li_copula_marginals_and_joint_defaults():
    p_curve = HazardCurve([3.0, 5.0], [0.02, 0.04])
    times = copula_default_times(p_curve, 0.4, 40000, n_names=2, rng=2)
    T = 5.0
    p = float(p_curve.default_probability(T))
    est = np.mean(times[:, 0] <= T)
    assert abs(est - p) < 4 * np.sqrt(p * (1 - p) / times.shape[0])
    joint = joint_default_probability(p, p, 0.4)
    est_joint = np.mean((times <= T).all(axis=1))
    assert abs(est_joint - joint) < 4 * np.sqrt(joint * (1 - joint) / times.shape[0])
    assert joint_default_probability(p, p, 0.0) == pytest.approx(p * p, rel=1e-6)
    assert 0 < default_correlation(p, p, 0.4) < 0.4
    # First-to-default probability of two names: P(tau1 <= T or tau2 <= T).
    assert nth_to_default_probability(times, 1, T) == pytest.approx(np.mean((times <= T).any(axis=1)))
    with pytest.raises(ValueError):
        copula_default_times(p_curve, 0.4, 10)


def test_basel_illustrative_risk_weights():
    """Basel II framework (BCBS 2006), illustrative IRB risk weights for corporates,
    LGD 45%, M = 2.5: 14.44% at PD 0.03% and 92.32% at PD 1%."""
    np.testing.assert_allclose(100 * np.asarray(basel_risk_weight([0.0003, 0.01])), [14.44, 92.32], atol=0.005)
    assert basel_risk_weight(0.01, scaling=1.06) == pytest.approx(1.06 * basel_risk_weight(0.01))


def test_basel_components():
    assert basel_correlation(1e-9) == pytest.approx(0.24, abs=1e-6)
    assert basel_correlation(0.99) == pytest.approx(0.12, abs=1e-6)
    assert basel_correlation(0.01, sales=5.0) == pytest.approx(basel_correlation(0.01) - 0.04)
    assert basel_correlation(0.01, "mortgage") == 0.15 and basel_correlation(0.01, "revolving") == 0.04
    assert basel_maturity_adjustment(0.01, 2.5) > 1.0
    assert basel_irb_capital(1.0) == 0.0
    # Retail has no maturity adjustment.
    assert basel_irb_capital(0.02, 0.5, 5.0, "retail") == basel_irb_capital(0.02, 0.5, 1.0, "retail")
    with pytest.raises(ValueError):
        basel_correlation(0.01, "bad")


def test_tranche_losses_add_up():
    p, rho, R = 0.05, 0.3, 0.4
    points = [0.0, 0.03, 0.07, 0.10, 0.15, 0.30, 1.0]
    els = [expected_tranche_loss(a, d, p, rho, R) * (d - a) for a, d in zip(points[:-1], points[1:], strict=True)]
    assert sum(els) == pytest.approx(p * (1 - R), rel=1e-9)
    assert tranche_loss(0.05, 0.03, 0.07) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        tranche_loss(0.1, 0.07, 0.03)


def test_finite_pool_tranche_converges_to_lhp_and_matches_monte_carlo():
    p, rho, R = 0.05, 0.3, 0.4
    lhp = expected_tranche_loss(0.03, 0.07, p, rho, R)
    assert expected_tranche_loss(0.03, 0.07, p, rho, R, n_names=5000) == pytest.approx(lhp, abs=2e-3)
    n_names, n = 100, 40000
    k = simulate_one_factor_defaults(p, rho, n_names, n, rng=4)
    x = tranche_loss((1 - R) * k / n_names, 0.03, 0.07)
    exact = expected_tranche_loss(0.03, 0.07, p, rho, R, n_names=n_names)
    assert abs(x.mean() - exact) < 4 * x.std(ddof=1) / np.sqrt(n)


def test_correlation_moves_risk_from_equity_to_senior():
    c = HazardCurve.flat(0.01)
    lo = [tranche_spread(a, d, c, 0.1, 5.0, 0.03).par_spread for a, d in ((0.0, 0.03), (0.07, 0.10))]
    hi = [tranche_spread(a, d, c, 0.5, 5.0, 0.03).par_spread for a, d in ((0.0, 0.03), (0.07, 0.10))]
    assert hi[0] < lo[0] and hi[1] > lo[1]
    # A [0, 100%] tranche with zero rates is close to the index spread lambda (1 - R).
    whole = tranche_spread(0.0, 1.0, c, 0.3, 5.0, 0.0, frequency=12)
    assert whole.par_spread == pytest.approx(0.006, rel=0.01)
