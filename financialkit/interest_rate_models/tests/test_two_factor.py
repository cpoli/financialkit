"""Tests for the G2++ two-factor Gaussian model."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from financialkit.constants import DEFAULT_RTOL
from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import (
    G2PlusPlusModel,
    HullWhiteModel,
    g2pp_swaption,
    jamshidian_swaption,
    monte_carlo_bond_prices,
    path_discount_factors,
    swaption_schedule,
)

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


def test_g2_returns_its_curve_and_validates_volatilities():
    assert MODEL.curve() is CURVE
    assert repr(MODEL) == "G2PlusPlusModel(a=0.6, sigma=0.012, b=0.06, eta=0.008, rho=-0.7)"
    with pytest.raises(ValueError, match="eta must be non-negative, got -0.008"):
        G2PlusPlusModel(CURVE, 0.6, 0.012, 0.06, -0.008, -0.7)


@pytest.mark.parametrize("expiry, tenor, strike", [(1.0, 5.0, 0.03), (3.0, 10.0, 0.05), (5.0, 2.0, 0.07)])
def test_g2pp_swaption_parity(expiry, tenor, strike):
    """Payer minus receiver is the forward payer swap, P(0, T) - sum c_i P(0, t_i)."""
    times = swaption_schedule(expiry, tenor)
    c = strike * np.diff(np.concatenate([[expiry], times]))
    c[-1] += 1.0
    forward = CURVE.discount(expiry) - c @ CURVE.discount(times)
    payer, receiver = g2pp_swaption(MODEL, expiry, tenor, strike), g2pp_swaption(MODEL, expiry, tenor, strike, payer=False)
    assert payer > 0 and receiver > 0
    assert payer - receiver == pytest.approx(forward, abs=1e-13)


def test_g2pp_swaption_matches_monte_carlo():
    T, K = 2.0, 0.045
    times, xy = MODEL.simulate_factors(T, 100, 40_000, rng=11)
    disc = path_discount_factors(times, xy.sum(axis=2) + MODEL.shift(times))[:, -1]
    pay = swaption_schedule(T, 5.0)
    c = K * np.diff(np.concatenate([[T], pay]))
    c[-1] += 1.0
    bonds = MODEL.zero_coupon_bond(pay[None, :], T, xy[:, -1][:, None, :])
    payoff = disc * np.maximum(1.0 - bonds @ c, 0.0)
    se = payoff.std(ddof=1) / np.sqrt(payoff.size)
    assert abs(payoff.mean() - g2pp_swaption(MODEL, T, 5.0, K)) < 4 * se


@pytest.mark.parametrize("payer", [True, False])
def test_g2pp_swaption_with_a_silent_factor_is_jamshidian(payer):
    # An independent factor with eta = 1e-7 moves the price by O(eta^2), about 1e-10 relative.
    hw = HullWhiteModel(CURVE, 0.3, 0.01)
    g2 = G2PlusPlusModel(CURVE, 0.3, 0.01, 0.1, 1e-7, 0.0)
    assert g2pp_swaption(g2, 3.0, 5.0, 0.04, payer=payer) == pytest.approx(jamshidian_swaption(hw, 3.0, 5.0, 0.04, payer=payer).price, rel=1e-8)


def test_g2pp_swaption_quadrature_converges_and_scales():
    coarse, fine = g2pp_swaption(MODEL, 5.0, 10.0, 0.05, n_nodes=24), g2pp_swaption(MODEL, 5.0, 10.0, 0.05, n_nodes=96)
    assert coarse == pytest.approx(fine, rel=1e-10)
    assert g2pp_swaption(MODEL, 5.0, 10.0, 0.05, notional=100.0) == pytest.approx(100.0 * g2pp_swaption(MODEL, 5.0, 10.0, 0.05), rel=DEFAULT_RTOL)


def test_g2pp_swaption_validation():
    with pytest.raises(ValueError, match="strike"):
        g2pp_swaption(MODEL, 1.0, 5.0, -0.01)
    with pytest.raises(ValueError, match="eta"):
        g2pp_swaption(G2PlusPlusModel(CURVE, 0.6, 0.012, 0.06, 0.0, -0.7), 1.0, 5.0, 0.04)
