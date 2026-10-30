"""Tests for OIS discounting with a separate forecasting curve."""

import numpy as np
import pytest

from financialkit.fixed_income import FlatCurve, MultiCurve, ZeroCurve, bootstrap_forecast_curve, bootstrap_multi_curve, bootstrap_par_curve, swap_rate

OIS_T = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
OIS_K = np.array([0.030, 0.031, 0.032, 0.034, 0.035, 0.036])
SWAP_T = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
SWAP_K = OIS_K + np.array([0.0030, 0.0028, 0.0026, 0.0024, 0.0022, 0.0020])


def test_bootstrap_reprices_par_ois_and_par_swaps():
    curves = bootstrap_multi_curve(OIS_T, OIS_K, SWAP_T, SWAP_K)
    assert swap_rate(curves.discount, OIS_T, freq=1) == pytest.approx(OIS_K, abs=1e-10)
    assert curves.swap_rate(SWAP_T) == pytest.approx(SWAP_K, abs=1e-10)
    for T, K in zip(SWAP_T, SWAP_K, strict=True):
        assert curves.swap_value(1e8, K, T).value == pytest.approx(0.0, abs=1e-1)


def test_single_curve_reduces_to_par_yield():
    curve = ZeroCurve([1.0, 5.0, 10.0], [0.02, 0.03, 0.035])
    single = MultiCurve(curve, curve)
    T = np.array([2.0, 5.0, 10.0])
    assert single.swap_rate(T, fixed_freq=2) == pytest.approx(curve.par_yield(T, 2), rel=1e-12)
    assert single.float_leg(5.0) == pytest.approx(1 - float(curve.discount(5.0)), rel=1e-12)
    forecast = bootstrap_forecast_curve(curve, T, curve.par_yield(T, 2), fixed_freq=2)
    assert MultiCurve(curve, forecast).swap_rate(T) == pytest.approx(curve.par_yield(T, 2), abs=1e-10)
    flat = FlatCurve(0.03)
    assert bootstrap_forecast_curve(flat, T, MultiCurve(flat, flat).swap_rate(T)).rates == pytest.approx(0.03, abs=1e-10)


def test_basis_lifts_forecast_forwards_above_ois():
    curves = bootstrap_multi_curve(OIS_T, OIS_K, SWAP_T, SWAP_K)
    ois = MultiCurve(curves.discount, curves.discount)
    t = np.arange(0.0, 9.76, 0.25)
    assert np.all(curves.forecast_rate(t, t + 0.25) > ois.forecast_rate(t, t + 0.25))


def test_off_market_swap_depends_on_discounting():
    """Par swaps reprice on either curve set; an off-market swap does not (Piterbarg, 2010)."""
    dual = bootstrap_multi_curve(OIS_T, OIS_K, SWAP_T, SWAP_K)
    libor = bootstrap_par_curve(SWAP_T, SWAP_K, freq=2)  # the pre-2008 single curve
    single = MultiCurve(libor, libor)
    assert single.swap_rate(SWAP_T) == pytest.approx(SWAP_K, abs=1e-10)
    v_dual = dual.swap_value(1e8, 0.05, 10.0, pay_fixed=False).value
    v_single = single.swap_value(1e8, 0.05, 10.0, pay_fixed=False).value
    assert v_dual > v_single > 0  # lower OIS rates discount the in-the-money annuity by less


def test_validation():
    curves = MultiCurve(FlatCurve(0.03), FlatCurve(0.03))
    with pytest.raises(ValueError):
        curves.float_leg(1.1)
    with pytest.raises(ValueError):
        curves.forecast_rate(1.0, 1.0)
    with pytest.raises(ValueError):
        bootstrap_forecast_curve(0.03, [1.0, 1.0], [0.03, 0.03])
