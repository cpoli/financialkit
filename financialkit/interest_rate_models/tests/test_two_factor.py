"""Tests for the G2++ two-factor Gaussian model."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from financialkit.constants import DEFAULT_RTOL
from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import G2PlusPlusModel, HullWhiteModel, monte_carlo_bond_prices, path_discount_factors

CURVE = NelsonSiegelCurve(0.05, -0.02, 0.015, 2.0)
MODEL = G2PlusPlusModel(CURVE, 0.6, 0.012, 0.06, 0.008, -0.7)


def test_g2pp_reprices_the_curve_exactly_and_by_simulation():
    T = np.array([1.0, 3.0, 8.0])
    assert np.allclose(MODEL.zero_coupon_bond(T), CURVE.discount(T), rtol=DEFAULT_RTOL)
    res = monte_carlo_bond_prices(MODEL, T, n_steps=160, n_paths=20_000, rng=4)
    assert np.all(np.abs(res.z_scores) < 4)


def test_g2pp_future_bond_prices_are_martingales_after_discounting():
    times, xy = MODEL.simulate_factors(2.0, 100, 20_000, rng=8)
    r = xy.sum(axis=2) + MODEL.shift(times)
    disc = path_discount_factors(times, r)[:, -1] * MODEL.zero_coupon_bond(7.0, t=2.0, state=xy[:, -1, :])
    assert abs(disc.mean() - CURVE.discount(7.0)) < 4 * disc.std(ddof=1) / np.sqrt(disc.size)


def test_g2pp_with_a_silent_second_factor_is_hull_white():
    g2 = G2PlusPlusModel(CURVE, 0.3, 0.01, 0.1, 0.0, 0.0)
    hw = HullWhiteModel(CURVE, 0.3, 0.01)
    assert g2.zero_coupon_bond(9.0, t=2.0, state=[0.004, 0.0]) == pytest.approx(float(hw.zero_coupon_bond(9.0, t=2.0, state=hw.shift(2.0) + 0.004)), rel=1e-6)
    assert g2.bond_option_volatility(2.0, 9.0) == pytest.approx(float(hw.bond_option_volatility(2.0, 9.0)), rel=1e-10)
    assert g2.yield_correlation(1.0, 20.0) == pytest.approx(1.0)


def test_g2pp_variance_function_matches_simulation():
    """V(0, T) is the variance of the integral of x + y."""
    times, xy = MODEL.simulate_factors(5.0, 250, 20_000, rng=6)
    I = trapezoid(xy.sum(axis=2), times, axis=1)
    assert I.var() == pytest.approx(float(MODEL.V(0.0, 5.0)), rel=0.04)


def test_negative_correlation_decorrelates_short_and_long_yields():
    rho = MODEL.yield_correlation(1.0, np.array([2.0, 10.0, 30.0]))
    assert np.all(np.diff(rho) < 0)
    assert rho[-1] < 0.6
    assert MODEL.yield_correlation(5.0, 5.0) == pytest.approx(1.0)


def test_validation():
    with pytest.raises(ValueError):
        G2PlusPlusModel(CURVE, 0.0, 0.01, 0.1, 0.01, 0.0)
    with pytest.raises(ValueError):
        G2PlusPlusModel(CURVE, 0.1, 0.01, 0.1, 0.01, 1.0)
