"""Tests for forward rates, bootstrapping, spline and parametric curve fits, and PCA."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_ATOL
from financialkit.fixed_income import (
    FlatCurve,
    NelsonSiegelCurve,
    SvenssonCurve,
    ZeroCurve,
    bond_cash_flows,
    bootstrap_par_curve,
    bootstrap_zero_curve,
    carleton_cooper_discount_factors,
    fit_mcculloch_spline,
    fit_nelson_siegel,
    fit_svensson,
    forward_rates,
    simulate_yield_changes,
    yield_curve_pca,
)


def test_lutz_forward_rates_hull_table_4_5():
    """Hull, 9th ed., Table 4.5: zero rates 3, 4, 4.6, 5, 5.3% give forwards -, 5.0, 5.8, 6.2, 6.5%."""
    f = forward_rates([1, 2, 3, 4, 5], [0.03, 0.04, 0.046, 0.05, 0.053])
    np.testing.assert_allclose(f, [0.03, 0.05, 0.058, 0.062, 0.065], atol=1e-12)


def test_forward_rates_average_back_to_zero_rates():
    t = np.array([0.5, 1, 2, 5, 10])
    r = np.array([0.01, 0.015, 0.02, 0.03, 0.035])
    f = forward_rates(t, r)
    np.testing.assert_allclose(np.cumsum(f * np.diff(np.r_[0, t])) / t, r, atol=1e-14)
    curve = ZeroCurve(t, r)
    np.testing.assert_allclose(curve.forward_rate(t[:-1], t[1:]), f[1:], atol=1e-14)


def test_bootstrap_hull_table_4_4():
    """Hull, 9th ed., Tables 4.3-4.4: zero rates 10.127, 10.469, 10.536, 10.681, 10.808%."""
    curve = bootstrap_zero_curve([0.25, 0.5, 1.0, 1.5, 2.0], [0, 0, 0, 0.08, 0.12], [97.5, 94.9, 90.0, 96.0, 101.6])
    np.testing.assert_allclose(100 * curve.rates, [10.127, 10.469, 10.536, 10.681, 10.808], atol=5e-4)


def test_bootstrap_reprices_input_bonds_with_interpolated_coupons():
    maturities = np.array([1.0, 2.0, 3.5, 5.0, 7.0, 10.0])
    coupons = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.042])
    true = ZeroCurve([0.5, 3.0, 10.0], [0.015, 0.03, 0.045])
    prices = [bond_cash_flows(100.0, c, T).present_value(true) for c, T in zip(coupons, maturities, strict=True)]
    curve = bootstrap_zero_curve(maturities, coupons, prices)
    for c, T, p in zip(coupons, maturities, prices, strict=True):
        assert bond_cash_flows(100.0, c, T).present_value(curve) == pytest.approx(p, abs=1e-8)


def test_par_curve_round_trip():
    tenors = np.array([1, 2, 3, 5, 7, 10, 20, 30.0])
    par = np.array([0.02, 0.024, 0.027, 0.031, 0.034, 0.037, 0.041, 0.042])
    curve = bootstrap_par_curve(tenors, par)
    np.testing.assert_allclose(curve.par_yield(tenors), par, atol=1e-9)
    flat = bootstrap_par_curve([1, 2, 5, 10], [0.05] * 4, freq=1)
    np.testing.assert_allclose(flat.rates, np.log(1.05), atol=1e-9)


def test_carleton_cooper_recovers_discount_factors_from_noisy_overdetermined_prices():
    rng = np.random.default_rng(3)
    times = np.arange(1, 6, dtype=float)
    d_true = np.exp(-0.04 * times)
    C = np.zeros((40, 5))
    for i in range(40):
        n = rng.integers(1, 6)
        C[i, :n] = 100 * rng.uniform(0.01, 0.08)
        C[i, n - 1] += 100
    prices = C @ d_true + rng.normal(0, 0.01, 40)
    res = carleton_cooper_discount_factors(times, C, prices)
    np.testing.assert_allclose(res.params, d_true, atol=5e-4)
    exact = carleton_cooper_discount_factors(
        times, C[:5] * 0 + np.triu(np.full((5, 5), 5.0)) + 100 * np.eye(5), (np.triu(np.full((5, 5), 5.0)) + 100 * np.eye(5)) @ d_true
    )
    np.testing.assert_allclose(exact.params, d_true, atol=1e-12)


def test_mcculloch_spline_fits_smooth_curve():
    true = NelsonSiegelCurve(0.045, -0.025, 0.01, 2.0)
    bonds = [bond_cash_flows(100.0, c, T) for c, T in zip(np.linspace(0.01, 0.05, 25), np.linspace(0.5, 20, 25), strict=True)]
    prices = [b.present_value(true) for b in bonds]
    for degree in (2, 3):
        res = fit_mcculloch_spline(bonds, prices, knots=[1, 2, 4, 7, 12], degree=degree)
        assert res.rmse < 0.02
        t = np.linspace(1, 20, 20)
        assert np.max(np.abs(res.curve.zero_rate(t) - true.zero_rate(t))) < 2e-4
        assert float(res.curve.discount(0.0)) == 1.0
    default = fit_mcculloch_spline(bonds, prices)
    assert default.curve.knots.size == 4 and default.rmse < 0.05


def test_nelson_siegel_limits_and_forward():
    curve = NelsonSiegelCurve(0.05, -0.02, 0.03, 1.5)
    assert float(curve.zero_rate(0.0)) == pytest.approx(0.03)
    assert float(curve.instantaneous_forward(0.0)) == pytest.approx(0.03)
    t = np.linspace(0.5, 20, 10)
    # closed-form forward vs the base class's central difference
    np.testing.assert_allclose(curve.instantaneous_forward(t), super(NelsonSiegelCurve, curve).instantaneous_forward(t), atol=1e-9)


def test_fit_nelson_siegel_and_svensson_recover_parameters():
    t = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30])
    ns = fit_nelson_siegel(t, NelsonSiegelCurve(0.05, -0.02, 0.015, 1.8).zero_rate(t))
    np.testing.assert_allclose(ns.params, [0.05, -0.02, 0.015, 1.8], atol=1e-6)
    true = SvenssonCurve(0.045, -0.015, -0.02, 0.03, 1.0, 6.0)
    sv = fit_svensson(t, true.zero_rate(t))
    assert sv.rmse < 1e-7
    assert ns.model == "nelson_siegel" and sv.model == "svensson"


def test_svensson_reduces_to_nelson_siegel():
    t = np.linspace(0.1, 30, 50)
    np.testing.assert_allclose(
        SvenssonCurve(0.04, -0.01, 0.02, 0.0, 2.0, 5.0).zero_rate(t), NelsonSiegelCurve(0.04, -0.01, 0.02, 2.0).zero_rate(t), atol=DEFAULT_ATOL
    )


def test_pca_level_slope_curvature():
    tenors = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    pca = yield_curve_pca(simulate_yield_changes(tenors, 3000, rng=1), tenors)
    ratio = pca.explained_variance_ratio
    assert ratio[0] > 0.7 and ratio[:3].sum() > 0.95
    assert np.all(pca.loadings[:, 0] > 0)
    assert pca.loadings[-1, 1] > pca.loadings[0, 1]
    np.testing.assert_allclose(pca.loadings.T @ pca.loadings, np.eye(tenors.size), atol=1e-10)
    assert np.all(np.diff(pca.eigenvalues) <= 1e-18)


def test_flat_curve_par_yield():
    assert FlatCurve(2 * np.log(1.03)).par_yield(10.0, freq=2) == pytest.approx(0.06, abs=1e-12)
