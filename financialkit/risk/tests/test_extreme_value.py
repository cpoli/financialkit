"""Tests for block maxima, GEV fits and peaks over threshold."""

import numpy as np
import pytest
from scipy import stats

from financialkit.constants import DEFAULT_RTOL
from financialkit.risk import GPDResult, block_maxima, fit_gev, fit_pot, gev_return_level, mean_excess, pot_es, pot_var


def test_block_maxima_of_losses():
    np.testing.assert_array_equal(block_maxima([1.0, -2.0, -0.5, 3.0, -4.0, 0.0, 9.0], 3), [2.0, 4.0])
    with pytest.raises(ValueError):
        block_maxima([1.0], 3)


def test_gev_fit_recovers_the_domain_of_attraction():
    rng = np.random.default_rng(0)
    n, m = 21, 1000
    t = fit_gev(-stats.t.rvs(3, size=n * m, random_state=rng), n)
    g = fit_gev(-rng.standard_normal(n * m), n)
    # Asymptotic sd of xi-hat is about 1/sqrt(m) times a factor near 1 for xi in [0, 0.5].
    se = 1.0 / np.sqrt(m)
    assert abs(t.xi - 1 / 3) < 4 * se
    assert abs(g.xi) < 4 * se
    assert t.maxima.size == m


def test_gev_return_level_inverts_the_cdf():
    fit = fit_gev(-np.random.default_rng(1).standard_t(4, 21 * 200), 21)
    k = np.array([10.0, 100.0])
    np.testing.assert_allclose(stats.genextreme.cdf(gev_return_level(fit, k), -fit.xi, fit.mu, fit.sigma), 1 - 1 / k, rtol=DEFAULT_RTOL)


def test_mean_excess_of_a_gpd_is_linear():
    xi, beta = 0.25, 1.0
    L = stats.genpareto.rvs(xi, scale=beta, size=400_000, random_state=2)
    u = np.array([0.0, 1.0, 2.0])
    me = mean_excess(-L, u)
    exact = (beta + xi * u) / (1 - xi)
    se = [np.std(L[L > ui] - ui) / np.sqrt(np.sum(L > ui)) for ui in u]
    assert np.all(np.abs(me - exact) < 4 * np.array(se))
    assert np.isnan(mean_excess(-L, [1e9])[0])


def test_pot_fit_and_formulas():
    rng = np.random.default_rng(3)
    xi, beta = 0.2, 0.5
    n = 50_000
    L = np.where(rng.random(n) < 0.1, 1.0 + stats.genpareto.rvs(xi, scale=beta, size=n, random_state=rng), rng.random(n))
    fit = fit_pot(-L, 1.0)
    # Smith (1987): the MLE of xi has asymptotic sd (1 + xi) / sqrt(N_u).
    assert abs(fit.xi - xi) < 4 * (1 + xi) / np.sqrt(fit.n_exceed)
    # The formulas themselves, with the true parameters and exceedance rate 10%.
    exact = GPDResult(xi=xi, beta=beta, threshold=1.0, n_exceed=1000, n_obs=10_000)
    q = 1.0 + stats.genpareto.ppf(1 - 0.001 / 0.1, xi, scale=beta)
    assert pot_var(exact, 0.999) == pytest.approx(q, rel=DEFAULT_RTOL)
    es_int = 1.0 + stats.genpareto.expect(lambda y: y, args=(xi,), scale=beta, lb=q - 1.0, conditional=True)
    assert pot_es(exact, 0.999) == pytest.approx(es_int, rel=1e-6)
    gumbel = GPDResult(xi=0.0, beta=beta, threshold=1.0, n_exceed=1000, n_obs=10_000)
    assert pot_var(gumbel, 0.999) == pytest.approx(1.0 + beta * np.log(100))


def test_pot_guards():
    with pytest.raises(ValueError):
        fit_pot(-np.arange(5.0), 1.0)
    with pytest.raises(ValueError):
        pot_es(GPDResult(xi=1.2, beta=1.0, threshold=0.0, n_exceed=10, n_obs=100), 0.99)
