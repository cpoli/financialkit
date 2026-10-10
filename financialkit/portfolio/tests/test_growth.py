"""Tests for the Kelly criterion, Merton's portfolio problem and CPPI."""

import numpy as np
import pytest
from scipy import optimize

from financialkit.integrators import gbm_paths
from financialkit.portfolio import (
    cppi,
    cppi_continuous,
    kelly_fraction,
    kelly_growth_rate,
    kelly_portfolio,
    kelly_wealth_paths,
    merton_certainty_equivalent,
    merton_consumption_rate,
    merton_fraction,
    merton_wealth_paths,
    tangency_portfolio,
)


def test_kelly_fraction_maximizes_growth():
    for p, b, a in [(0.6, 1.0, 1.0), (0.3, 3.0, 1.0), (0.55, 1.0, 0.5)]:
        f = kelly_fraction(p, b, a)
        best = optimize.minimize_scalar(lambda x, p=p, b=b, a=a: -kelly_growth_rate(x, p, b, a), bounds=(0, 0.999 / a), method="bounded")
        assert f == pytest.approx(best.x, abs=1e-5)
    # Kelly (1956): with even odds the growth rate is the channel capacity 1 - H(p) in nats.
    p = 0.6
    assert kelly_growth_rate(2 * p - 1, p) == pytest.approx(np.log(2) + p * np.log(p) + (1 - p) * np.log(1 - p))
    # Betting twice Kelly gives zero growth to second order; three times is ruinous.
    assert kelly_growth_rate(3 * 0.2, 0.6) < 0


def test_kelly_wealth_paths_growth_rate():
    f, p, n = 0.2, 0.6, 400
    W = kelly_wealth_paths(f, p, n, 2000, rng=0)
    rates = np.log(W[:, -1]) / n
    assert abs(rates.mean() - kelly_growth_rate(f, p)) < 4 * rates.std(ddof=1) / np.sqrt(rates.size)


def test_kelly_portfolio_is_log_utility_tangency_leverage():
    mu, cov, r = np.array([0.08, 0.1]), np.array([[0.04, 0.01], [0.01, 0.09]]), 0.02
    k = kelly_portfolio(mu, cov, r)
    np.testing.assert_allclose(k / k.sum(), tangency_portfolio(mu, cov, r).weights)
    assert merton_fraction(0.08, 0.02, 0.2, 1.0) == pytest.approx(kelly_portfolio([0.08], [[0.04]], 0.02)[0])


def test_merton_fraction_maximizes_certainty_equivalent():
    mu, r, sigma, gamma = 0.08, 0.02, 0.2, 3.0
    pis = np.linspace(-1, 2, 3001)
    ce = merton_certainty_equivalent(pis, mu, r, sigma, gamma)
    assert pis[np.argmax(ce)] == pytest.approx(merton_fraction(mu, r, sigma, gamma), abs=1e-3)


def test_merton_wealth_expected_utility_by_monte_carlo():
    """E[W_T^(1-g)] = exp((1-g) CE T) for the constant-mix strategy."""
    mu, r, sigma, gamma, T = 0.08, 0.02, 0.2, 3.0, 5.0
    pi = merton_fraction(mu, r, sigma, gamma)
    _, W = merton_wealth_paths(pi, mu, r, sigma, T, 1, 200_000, rng=0)
    u = W[:, -1] ** (1 - gamma)
    exact = np.exp((1 - gamma) * merton_certainty_equivalent(pi, mu, r, sigma, gamma) * T)
    assert abs(u.mean() - exact) < 4 * u.std(ddof=1) / np.sqrt(u.size)


def test_merton_consumption_rate():
    assert merton_consumption_rate(0.08, 0.02, 0.2, 1.0, 0.03) == pytest.approx(0.03)
    # Merton (1969), eq. (34): nu = rho/g - (1-g)/g (r + theta^2 / (2 g)).
    theta = 0.3
    assert merton_consumption_rate(0.08, 0.02, 0.2, 2.0, 0.04) == pytest.approx(0.04 / 2 + 0.5 * (0.02 + theta**2 / 4))
    with pytest.raises(ValueError):
        merton_consumption_rate(0.08, 0.02, 0.2, 0.5, 0.0)


def test_cppi_converges_to_black_perold_closed_form():
    sigma, r = 0.25, 0.03
    errs = []
    for n_steps in [250, 2500]:
        rng = np.random.default_rng(1)
        times, S = gbm_paths(100.0, 0.07, sigma, 1.0, n_steps, 50, rng=rng)
        disc = cppi(S, times, floor=0.8, multiplier=4.0, r=r)
        exact = cppi_continuous(S, times, sigma, floor=0.8, multiplier=4.0, r=r)
        errs.append(np.abs(disc.wealth[:, -1] - exact[:, -1]).mean())
    assert errs[1] < errs[0] / 2
    assert errs[1] < 2e-3


def test_cppi_without_leverage_never_breaches_with_m_one():
    times, S = gbm_paths(100.0, 0.05, 0.6, 1.0, 12, 500, rng=2)
    assert not cppi(S, times, floor=0.9, multiplier=1.0).breached.any()
    res = cppi(S, times, floor=0.9, multiplier=8.0, max_leverage=1.0)
    assert np.all(res.exposure <= res.wealth[:, :-1] + 1e-12)
    assert res.breached.any()
    with pytest.raises(ValueError):
        cppi(S[:, :-1], times)
