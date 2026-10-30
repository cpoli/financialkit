"""Heston vega in the initial variance, one bump_greek call each for the Fourier and Monte Carlo pricers."""

import numpy as np
import pytest

from financialkit.options import black_scholes_greeks, bump_greek
from financialkit.stochastic_volatility import HestonModel, heston_mc_price, heston_price


def test_heston_v0_vega_in_the_deterministic_variance_limit():
    """As xi -> 0 the variance is deterministic, v_t = theta + (v0 - theta) e^{-kappa t},
    so the price is Black-Scholes at the mean variance theta + (v0 - theta) a with
    a = (1 - e^{-kappa T}) / (kappa T), and dC/dv0 = vega(sigma_bar) a / (2 sigma_bar).
    With xi = 1e-3 the vol-of-vol correction is O(xi^2) = 1e-6 relative."""
    v0, kappa, theta, T = 0.05, 2.0, 0.04, 1.0
    m = HestonModel(v0=v0, kappa=kappa, theta=theta, xi=1e-3, rho=0.0)
    vega_v0 = bump_greek(heston_price, "model.v0", m, 100.0, 100.0, T, 0.03)
    a = (1.0 - np.exp(-kappa * T)) / (kappa * T)
    sigma_bar = np.sqrt(theta + (v0 - theta) * a)
    exact = black_scholes_greeks(100.0, 100.0, T, 0.03, sigma_bar).vega * a / (2.0 * sigma_bar)
    assert vega_v0 == pytest.approx(exact, rel=1e-5)
    assert m.v0 == v0


def test_heston_mc_v0_vega_matches_the_fourier_vega():
    """Ten independent common-random-number estimates of dC/dv0 under QE, whose mean
    is within four standard errors of the Fourier value."""
    m = HestonModel(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
    exact = bump_greek(heston_price, "model.v0", m, 100.0, 100.0, 1.0, 0.03)
    est = [bump_greek(heston_mc_price, "model.v0", m, 100.0, 100.0, 1.0, 0.03, n_steps=20, n_paths=20_000, rng=seed, h=1e-3) for seed in range(10)]
    assert abs(np.mean(est) - exact) < 4 * np.std(est, ddof=1) / np.sqrt(len(est))
