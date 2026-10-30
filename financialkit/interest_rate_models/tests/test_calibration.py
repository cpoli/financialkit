"""Tests for calibrating Hull-White and G2++ to Black cap and swaption volatilities."""

import numpy as np
import pytest

from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import (
    G2PlusPlusModel,
    HullWhiteModel,
    black_swaption_price,
    cap_floor_price,
    forward_swap_rate,
    g2pp_calibrate,
    g2pp_swaption,
    hull_white_calibrate,
    jamshidian_swaption,
    model_cap_floor_price,
    model_implied_vols,
)

CURVE = NelsonSiegelCurve(0.04, -0.015, 0.01, 2.0)
CAP_MATURITIES = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
EXPIRIES = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
TENORS = np.array([5.0, 5.0, 2.0, 5.0, 3.0])


def test_model_implied_vols_reprice_the_model():
    hw = HullWhiteModel(CURVE, 0.08, 0.009)
    caps = model_implied_vols(hw, CAP_MATURITIES, strikes=0.035)
    for T, v in zip(CAP_MATURITIES, caps, strict=True):
        assert cap_floor_price(CURVE, 0.035, T, v).price == pytest.approx(model_cap_floor_price(hw, 0.035, T).price, rel=1e-10)
    swaptions = model_implied_vols(hw, EXPIRIES, "swaption", TENORS, strikes=0.04)
    for T, t, v in zip(EXPIRIES, TENORS, swaptions, strict=True):
        assert black_swaption_price(CURVE, T, t, 0.04, v) == pytest.approx(jamshidian_swaption(hw, T, t, 0.04).price, rel=1e-10)


@pytest.mark.parametrize("kind", ["cap", "swaption"])
def test_hull_white_round_trip(kind):
    """Black vols implied by a known Hull-White model are fitted back to its (a, sigma)."""
    hw = HullWhiteModel(CURVE, 0.08, 0.009)
    expiries, tenors = (CAP_MATURITIES, None) if kind == "cap" else (EXPIRIES, TENORS)
    vols = model_implied_vols(hw, expiries, kind, tenors)
    fit = hull_white_calibrate(CURVE, vols, expiries, kind, tenors)
    assert fit.converged and fit.kind == kind
    assert fit.params == pytest.approx({"a": 0.08, "sigma": 0.009}, rel=1e-6)
    assert fit.rmse < 1e-8  # vol errors far below a hundredth of a basis point
    assert np.allclose(fit.model_prices, fit.market_prices, rtol=1e-8)
    assert isinstance(fit.model, HullWhiteModel)


def test_hull_white_round_trip_off_the_money_caps():
    hw = HullWhiteModel(CURVE, 0.15, 0.011)
    strikes = np.linspace(0.02, 0.06, CAP_MATURITIES.size)
    vols = model_implied_vols(hw, CAP_MATURITIES, strikes=strikes)
    fit = hull_white_calibrate(CURVE, vols, CAP_MATURITIES, strikes=strikes, x0=(0.05, 0.02))
    assert fit.params == pytest.approx({"a": 0.15, "sigma": 0.011}, rel=1e-6)
    assert np.array_equal(fit.strikes, strikes)


def test_g2pp_round_trip_on_caps():
    """A 20-maturity cap strip from a known G2++ model is refitted to within 1e-8 in vol."""
    g2 = G2PlusPlusModel(CURVE, 0.6, 0.012, 0.06, 0.007, -0.6)
    maturities = np.arange(1.0, 21.0)
    vols = model_implied_vols(g2, maturities)
    fit = g2pp_calibrate(CURVE, vols, maturities)
    assert fit.converged and fit.rmse < 1e-8
    assert np.allclose(fit.model_prices, fit.market_prices, rtol=1e-8)
    assert set(fit.params) == {"a", "sigma", "b", "eta", "rho"}


def test_g2pp_round_trip_on_swaptions():
    """A swaption grid from a known G2++ model is refitted to its parameters.

    Tenors from 1 to 10 years see the correlation of short and long rates,
    so unlike a cap strip the grid identifies all five parameters.
    """
    g2 = G2PlusPlusModel(CURVE, 0.6, 0.012, 0.06, 0.007, -0.6)
    expiries, tenors = (g.ravel() for g in np.meshgrid([1.0, 2.0, 5.0, 10.0], [1.0, 5.0, 10.0]))
    vols = model_implied_vols(g2, expiries, "swaption", tenors)
    for T, t, v in zip(expiries, tenors, vols, strict=True):
        K = forward_swap_rate(CURVE, T, t)
        # brentq stops at DEFAULT_ATOL in vol, a few 1e-10 relative in price.
        assert black_swaption_price(CURVE, T, t, K, v) == pytest.approx(g2pp_swaption(g2, T, t, K), rel=1e-8)
    fit = g2pp_calibrate(CURVE, vols, expiries, kind="swaption", tenors=tenors)
    assert fit.converged and fit.kind == "swaption" and fit.rmse < 1e-8
    assert fit.params == pytest.approx({"a": 0.6, "sigma": 0.012, "b": 0.06, "eta": 0.007, "rho": -0.6}, rel=1e-4)
    assert isinstance(fit.model, G2PlusPlusModel)


def test_calibration_validation():
    with pytest.raises(ValueError, match="kind"):
        hull_white_calibrate(CURVE, [0.2, 0.2], [1.0, 2.0], kind="floor")
    with pytest.raises(ValueError, match="tenors"):
        hull_white_calibrate(CURVE, [0.2, 0.2], [1.0, 2.0], kind="swaption")
    with pytest.raises(ValueError, match="at least 2"):
        hull_white_calibrate(CURVE, [0.2], [1.0])
    with pytest.raises(ValueError, match="positive"):
        hull_white_calibrate(CURVE, [0.2, -0.1], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least 5"):
        g2pp_calibrate(CURVE, [0.2] * 4, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="tenors"):
        g2pp_calibrate(CURVE, [0.2] * 5, [1.0, 2.0, 3.0, 4.0, 5.0], kind="swaption")


def test_model_implied_vols_nan_when_black_cannot_match():
    # Gaussian rates with 10% absolute volatility make a cap worth more than
    # Black's lognormal price at 500% volatility.
    hw = HullWhiteModel(CURVE, 0.0, 0.1)
    assert np.isnan(model_implied_vols(hw, [5.0])[0])
