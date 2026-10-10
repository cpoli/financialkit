"""Tests for SABR, SVI and Lee's moment formula."""

import numpy as np
import pytest

from financialkit.stochastic_volatility import (
    HestonModel,
    KouModel,
    VarianceGammaModel,
    durrleman_g,
    lee_wing_slope,
    sabr_calibrate,
    sabr_implied_vol,
    svi_fit,
    svi_total_variance,
)


def test_sabr_atm_formula():
    """Hagan et al. (2002), eq. (2.18): the at-the-money limit of the expansion."""
    F, T, a, b, rho, nu = 0.03, 2.0, 0.02, 0.5, -0.3, 0.4
    fb = F ** (1 - b)
    atm = a / fb * (1 + ((1 - b) ** 2 * a**2 / (24 * fb**2) + rho * b * nu * a / (4 * fb) + (2 - 3 * rho**2) * nu**2 / 24) * T)
    assert sabr_implied_vol(F, F, T, a, b, rho, nu) == pytest.approx(atm, rel=1e-14)
    # Continuous through the money.
    assert sabr_implied_vol(F, F * (1 + 1e-7), T, a, b, rho, nu) == pytest.approx(atm, rel=1e-6)


def test_sabr_parameters_shape_the_smile():
    K = np.array([80.0, 100.0, 120.0])
    neg = np.asarray(sabr_implied_vol(100.0, K, 1.0, 0.2, 1.0, -0.5, 0.5))
    curv = np.asarray(sabr_implied_vol(100.0, K, 1.0, 0.2, 1.0, 0.0, 0.8))
    assert neg[0] > neg[1] > neg[2]
    assert curv[0] > curv[1] < curv[2]


def test_sabr_calibration_recovers_parameters():
    K = np.linspace(0.02, 0.05, 15)
    vols = sabr_implied_vol(0.03, K, 5.0, 0.025, 0.5, -0.25, 0.35)
    fit = sabr_calibrate(0.03, K, 5.0, vols, beta=0.5)
    assert (fit.alpha, fit.rho, fit.nu) == pytest.approx((0.025, -0.25, 0.35), abs=1e-6)
    assert fit.rmse < 1e-10
    assert np.asarray(fit.implied_vol(K)) == pytest.approx(vols, abs=1e-10)


def test_svi_fits_heston_smile_without_arbitrage():
    """Gatheral (2004): SVI is the large-maturity limit of Heston and fits its smile closely."""
    m = HestonModel(0.04, 1.5, 0.04, 0.6, -0.7)
    k = np.linspace(-0.5, 0.4, 19)
    iv = np.asarray(m.implied_vol(100.0, 100.0 * np.exp(k), 1.0, 0.0))
    fit = svi_fit(k, iv**2, 1.0)
    assert fit.rmse < 2e-4
    assert np.asarray(fit.implied_vol(k)) == pytest.approx(iv, abs=2e-3)
    assert np.min(durrleman_g(np.linspace(-1.5, 1.5, 61), fit.a, fit.b, fit.rho, fit.m, fit.sigma)) > 0


def test_svi_wings_are_linear():
    a, b, rho, m, s = 0.02, 0.1, -0.4, 0.05, 0.2
    k = np.array([50.0, 60.0])
    slope = np.diff(np.asarray(svi_total_variance(k, a, b, rho, m, s)))[0] / 10
    assert slope == pytest.approx(b * (1 + rho), rel=1e-4)


def test_durrleman_detects_butterfly_arbitrage():
    """Wings steeper than Lee's bound of 2 make the implied density negative somewhere."""
    k = np.linspace(-3, 3, 121)
    assert np.min(durrleman_g(k, 0.0, 1.5, 0.9, 0.0, 0.05)) < 0


def test_lee_wing_slope_limits_and_kou():
    assert lee_wing_slope(0.0) == 2.0
    assert lee_wing_slope(np.inf) == 0.0
    assert lee_wing_slope(1.0) == pytest.approx(2 - 4 * (np.sqrt(2) - 1))
    # Kou with eta1 = 3: E[S^{1+p}] is finite for p < 2, and the right wing w(k)/k approaches the bound from above.
    m = KouModel(0.1, 1.0, 0.5, 3.0, 2.0)
    p, q = m.critical_moments()
    k = np.array([4.0, 8.0])
    w = np.asarray(m.implied_vol(100.0, 100.0 * np.exp(k), 1.0, 0.0)) ** 2
    ratios = w / k
    assert ratios[0] > ratios[1] > lee_wing_slope(p)


def test_variance_gamma_lee_slope_in_range():
    p, q = VarianceGammaModel(0.2, 0.5, -0.2).critical_moments()
    assert 0 < lee_wing_slope(p) < lee_wing_slope(q) < 2
