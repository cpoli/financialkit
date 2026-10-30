"""Tests for CEV and Dupire local volatility."""

import numpy as np
import pytest

from financialkit.integrators import euler_maruyama, njit
from financialkit.options import black_scholes_price, implied_volatility
from financialkit.stochastic_volatility import LocalVolSurface, cev_local_vol, cev_price, dupire_local_vol, local_vol_paths, sabr_implied_vol


@njit
def _cev_drift(x, t, p):
    return p[0] * x


@njit
def _cev_diffusion(x, t, p):
    return p[1] * np.maximum(x, 0.0) ** p[2]


@pytest.mark.parametrize("beta", [0.5, 1.5])
def test_cev_matches_monte_carlo(beta):
    sigma = 0.2 * 100 ** (1 - beta)
    _, S = euler_maruyama(_cev_drift, _cev_diffusion, 100.0, 1.0, 100, 50_000, params=[0.05, sigma, beta], rng=3)
    pay = np.exp(-0.05) * np.maximum(S[:, -1] - 110.0, 0.0)
    # Euler bias at 100 steps is well below the statistical error for these parameters.
    assert abs(pay.mean() - cev_price(100.0, 110.0, 1.0, 0.05, sigma, beta)) < 4 * pay.std() / np.sqrt(pay.size)


@pytest.mark.parametrize("beta", [0.4, 1.6])
def test_cev_put_call_parity_and_dividends(beta):
    K = np.array([80.0, 100.0, 120.0])
    c = np.asarray(cev_price(100.0, K, 0.75, 0.04, 0.2 * 100 ** (1 - beta), beta, q=0.02))
    p = np.asarray(cev_price(100.0, K, 0.75, 0.04, 0.2 * 100 ** (1 - beta), beta, q=0.02, option_type="put"))
    assert c - p == pytest.approx(100 * np.exp(-0.015) - K * np.exp(-0.03), abs=1e-10)


def test_cev_tends_to_black_scholes_as_beta_tends_to_one():
    K = np.array([90.0, 100.0, 110.0])
    near = np.asarray(cev_price(100.0, K, 1.0, 0.03, 0.2 * 100**0.001, 0.999))
    assert near == pytest.approx(black_scholes_price(100.0, K, 1.0, 0.03, 0.2), abs=5e-3)


def test_cev_matches_hagan_backbone():
    """SABR with nu = 0 is CEV; Hagan's expansion matches the exact CEV implied vol to O(eps^2)."""
    beta, a = 0.5, 2.0
    K = np.array([70.0, 100.0, 130.0])
    exact = np.asarray(implied_volatility(cev_price(100.0, K, 1.0, 0.0, a, beta), 100.0, K, 1.0, 0.0))
    assert np.asarray(sabr_implied_vol(100.0, K, 1.0, a, beta, 0.0, 0.0)) == pytest.approx(exact, abs=1e-5)


def test_cev_rejects_lognormal_beta():
    with pytest.raises(ValueError):
        cev_price(100.0, 100.0, 1.0, 0.0, 0.2, 1.0)


def test_cev_local_vol():
    assert cev_local_vol(25.0, 2.0, 0.5) == pytest.approx(0.4)


def test_dupire_recovers_cev_local_vol():
    beta, sigma, r = 0.5, 2.5, 0.03
    K = np.linspace(60, 160, 101)
    T = np.linspace(0.5, 1.5, 41)
    C = cev_price(100.0, K[None, :], T[:, None], r, sigma, beta)
    lv = dupire_local_vol(K, T, C, r)
    # Second-order differences on a 1.0 x 0.025 grid: errors of order 1e-4 away from the strike edges.
    assert np.abs(lv.vol[:, 5:-5] - np.asarray(cev_local_vol(K[5:-5], sigma, beta))).max() < 2e-3


def test_dupire_with_dividends_on_black_scholes():
    K = np.linspace(70, 140, 71)
    T = np.linspace(0.5, 1.0, 11)
    C = black_scholes_price(100.0, K[None, :], T[:, None], 0.04, 0.3, 0.02)
    lv = dupire_local_vol(K, T, C, 0.04, 0.02)
    assert lv.vol[:, 3:-3] == pytest.approx(0.3, abs=1e-3)
    with pytest.raises(ValueError):
        dupire_local_vol(K, T, C[:, :-1], 0.04)


def test_local_vol_surface_interpolation():
    s = LocalVolSurface(np.array([80.0, 120.0]), np.array([0.0, 1.0]), np.array([[0.3, 0.1], [0.5, 0.3]]))
    assert s(100.0, 0.5) == pytest.approx(0.3)
    assert s(50.0, 2.0) == pytest.approx(0.5)


def test_local_vol_paths_reprice_cev():
    beta, sigma, r = 0.5, 2.5, 0.03
    K = np.linspace(40, 220, 181)
    T = np.linspace(0.05, 1.0, 39)
    lv = dupire_local_vol(K, T, cev_price(100.0, K[None, :], T[:, None], r, sigma, beta), r)
    # The true surface: removes the Dupire error so the test isolates the simulator.
    exact = LocalVolSurface(K, T, np.broadcast_to(np.asarray(cev_local_vol(K, sigma, beta)), lv.vol.shape).copy())
    _, S = local_vol_paths(exact, 100.0, 1.0, r, n_steps=100, n_paths=40_000, rng=5)
    pay = np.exp(-r) * np.maximum(S[:, -1] - 100.0, 0.0)
    assert abs(pay.mean() - cev_price(100.0, 100.0, 1.0, r, sigma, beta)) < 4 * pay.std() / np.sqrt(pay.size)


def test_local_vol_paths_fill_nan_nodes():
    s = LocalVolSurface(np.array([80.0, 100.0, 120.0]), np.array([0.0, 1.0]), np.array([[np.nan, 0.2, 0.2], [0.2, 0.2, np.nan]]))
    _, S = local_vol_paths(s, 100.0, 1.0, 0.0, n_steps=10, n_paths=100, rng=0)
    assert np.isfinite(S).all()
    bad = LocalVolSurface(np.array([80.0, 120.0]), np.array([0.0, 1.0]), np.array([[np.nan, np.nan], [0.2, 0.2]]))
    with pytest.raises(ValueError):
        local_vol_paths(bad, 100.0, 1.0, 0.0, n_steps=10, n_paths=10)


def test_local_vol_paths_hold_the_last_maturity_flat_beyond_it():
    """Vol 0.1 -> 0.3 linearly over [0, 0.5], then 0.3: total variance 0.0217 + 0.045 = 1/15 by T = 1."""
    s = LocalVolSurface(np.array([50.0, 200.0]), np.array([0.0, 0.5]), np.array([[0.1, 0.1], [0.3, 0.3]]))
    _, S = local_vol_paths(s, 100.0, 1.0, 0.03, n_steps=200, n_paths=40_000, rng=6)
    pay = np.exp(-0.03) * np.maximum(S[:, -1] - 100.0, 0.0)
    exact = black_scholes_price(100.0, 100.0, 1.0, 0.03, np.sqrt(1.0 / 15.0))
    assert abs(pay.mean() - exact) < 4 * pay.std() / np.sqrt(pay.size)
