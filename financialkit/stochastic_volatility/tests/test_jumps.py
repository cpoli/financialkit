"""Tests for Merton, variance gamma, Kou and Bates."""

import numpy as np
import pytest
from scipy.integrate import quad

from financialkit.options import black_scholes_price
from financialkit.stochastic_volatility import BatesModel, HestonModel, KouModel, MertonJumpModel, VarianceGammaModel, lewis_price, merton_jump_price

K = np.array([80.0, 100.0, 120.0])


def test_merton_series_matches_fourier():
    m = MertonJumpModel(0.2, 1.0, -0.1, 0.15)
    for option_type in ("call", "put"):
        series = np.asarray(merton_jump_price(100.0, K, 0.5, 0.05, m, q=0.01, option_type=option_type))
        assert series == pytest.approx(np.asarray(m.price(100.0, K, 0.5, 0.05, 0.01, option_type)), abs=1e-9)


def test_merton_without_jumps_is_black_scholes():
    m = MertonJumpModel(0.25, 0.0, 0.0, 0.1)
    assert np.asarray(merton_jump_price(100.0, K, 1.0, 0.03, m)) == pytest.approx(black_scholes_price(100.0, K, 1.0, 0.03, 0.25))


def test_merton_monte_carlo():
    m = MertonJumpModel(0.2, 1.0, -0.1, 0.15)
    x = m.sample_terminal(0.5, 200_000, rng=1)
    pay = np.exp(-0.025) * np.maximum(100 * np.exp(0.025) * np.exp(x) - 100.0, 0.0)
    assert abs(pay.mean() - merton_jump_price(100.0, 100.0, 0.5, 0.05, m)) < 4 * pay.std() / np.sqrt(pay.size)


def test_kou_published_value():
    """Kou (2002), Section 5 (Table 1): S = 100, K = 98, T = 0.5, r = 0.05, sigma = 0.16, lambda = 1,
    p = 0.4, eta1 = 10, eta2 = 5 gives a call worth 9.14732."""
    m = KouModel(0.16, 1.0, 0.4, 10.0, 5.0)
    assert lewis_price(m, 100.0, 98.0, 0.5, 0.05) == pytest.approx(9.14732, abs=5e-6)


def test_kou_monte_carlo():
    m = KouModel(0.16, 1.0, 0.4, 10.0, 5.0)
    x = m.sample_terminal(0.5, 200_000, rng=2)
    pay = np.exp(-0.025) * np.maximum(100 * np.exp(0.025) * np.exp(x) - 98.0, 0.0)
    assert abs(pay.mean() - m.price(100.0, 98.0, 0.5, 0.05)) < 4 * pay.std() / np.sqrt(pay.size)


def test_variance_gamma_density_and_sampling():
    m = VarianceGammaModel(0.12, 0.2, -0.14)
    assert quad(lambda x: float(m.pdf(x, 1.0)), -3, 3, points=[m.omega], limit=200)[0] == pytest.approx(1.0, abs=1e-8)
    x = m.sample_terminal(1.0, 200_000, rng=3)
    assert abs(np.exp(x).mean() - 1.0) < 4 * np.exp(x).std() / np.sqrt(x.size)
    pay = np.exp(-0.1) * np.maximum(100 * np.exp(0.1) * np.exp(x) - 90.0, 0.0)
    assert abs(pay.mean() - 19.099354724) < 4 * pay.std() / np.sqrt(pay.size)


def test_symmetric_variance_gamma_has_symmetric_smile():
    """Madan-Seneta (theta = 0): the log-return density is symmetric about its mode, so the smile is
    symmetric in log-moneyness up to the martingale drift; theta < 0 tilts it into a skew."""
    F = 100.0
    k = np.array([-0.3, 0.3])
    sym = np.asarray(VarianceGammaModel(0.2, 0.5, 0.0).implied_vol(100.0, F * np.exp(k), 1.0, 0.0))
    skew = np.asarray(VarianceGammaModel(0.2, 0.5, -0.3).implied_vol(100.0, F * np.exp(k), 1.0, 0.0))
    assert skew[0] - skew[1] > 5 * abs(sym[0] - sym[1])


def test_variance_gamma_critical_moments():
    m = VarianceGammaModel(0.2, 0.5, -0.2)
    p, q = m.critical_moments()
    base = lambda s: 1 - m.theta * m.nu * s - 0.5 * m.sigma**2 * m.nu * s**2  # noqa: E731
    assert base(1 + p) == pytest.approx(0.0, abs=1e-12)
    assert base(-q) == pytest.approx(0.0, abs=1e-12)


def test_variance_gamma_rejects_infinite_mean():
    with pytest.raises(ValueError):
        VarianceGammaModel(1.5, 2.0, 0.5)


def test_bates_combines_heston_and_merton():
    b = BatesModel(0.04, 1.5, 0.04, 0.4, -0.6, 0.7, -0.1, 0.1)
    h = HestonModel(0.04, 1.5, 0.04, 0.4, -0.6)
    j = MertonJumpModel(0.0, 0.7, -0.1, 0.1)
    u = np.linspace(-5, 5, 11)
    assert b.char_func(u, 1.0) == pytest.approx(h.char_func(u, 1.0) * j.char_func(u, 1.0))
    # Jumps fatten the short-dated left wing.
    assert b.implied_vol(100.0, 80.0, 0.25, 0.0) > h.implied_vol(100.0, 80.0, 0.25, 0.0) + 0.02


def test_kou_rejects_bad_parameters():
    with pytest.raises(ValueError):
        KouModel(0.1, 1.0, 0.5, 0.9, 5.0)
    with pytest.raises(ValueError):
        KouModel(0.1, 1.0, 1.5, 3.0, 5.0)
