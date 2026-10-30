"""Tests for the IRR, multiple IRRs, MIRR and the NPV profile."""

import warnings

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.time_value import MultipleIRRWarning, irr, irr_all, mirr, npv, npv_profile, sign_changes


def test_irr_zeroes_npv():
    cf = [-1000.0, 300.0, 400.0, 500.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = irr(cf)
    assert abs(npv(r, cf)) < 1e-6
    assert irr_all(cf) == pytest.approx([r], rel=1e-9)


def test_irr_two_period_closed_form():
    """-100 + 60 x + 60 x^2 = 0 with x = 1 / (1 + r)."""
    x = (-60 + np.sqrt(60**2 + 4 * 60 * 100)) / 120
    assert irr([-100.0, 60.0, 60.0]) == pytest.approx(1 / x - 1, rel=DEFAULT_RTOL)


def test_irr_with_times_and_continuous_compounding():
    """-100 at 0 and 110 at t = 2 years: continuous IRR = ln(1.1) / 2."""
    assert irr([-100.0, 110.0], times=[0.0, 2.0], compounding=None) == pytest.approx(np.log(1.1) / 2, rel=DEFAULT_RTOL)


def test_lorie_savage_pump_has_two_irrs():
    """Lorie and Savage (1955), *Journal of Business* 28, p. 237: the pump
    project -1,600, +10,000, -10,000 has IRRs of 25% and 400%."""
    cf = [-1600.0, 10_000.0, -10_000.0]
    assert sign_changes(cf) == 2
    np.testing.assert_allclose(irr_all(cf), [0.25, 4.0], rtol=1e-9)
    with pytest.warns(MultipleIRRWarning):
        assert irr(cf) == pytest.approx(0.25, rel=DEFAULT_RTOL)


def test_irr_raises_without_sign_change():
    with pytest.raises(ValueError):
        irr([100.0, 50.0])
    assert irr_all([100.0, 50.0]).size == 0


def test_mirr_excel_reference_values():
    """Microsoft Excel's MIRR documentation example: cash flows -120,000,
    39,000, 30,000, 21,000, 37,000, 46,000, finance rate 10%: MIRR is
    12.61% at a 12% reinvestment rate and 13.48% at 14%; the first three
    years alone give -4.80%."""
    cf = [-120_000, 39_000, 30_000, 21_000, 37_000, 46_000]
    assert round(mirr(cf, 0.10, 0.12), 4) == 0.1261
    assert round(mirr(cf, 0.10, 0.14), 4) == 0.1348
    assert round(mirr(cf[:4], 0.10, 0.12), 4) == -0.0480


def test_mirr_equals_irr_when_rates_equal_irr():
    cf = [-1000.0, 300.0, 400.0, 500.0]
    r = irr(cf)
    assert mirr(cf, r, r) == pytest.approx(r, rel=1e-8)
    with pytest.raises(ValueError):
        mirr([100.0, 100.0], 0.1, 0.1)


def test_npv_profile_crosses_zero_at_irrs():
    profile = npv_profile([-1600.0, 10_000.0, -10_000.0], np.linspace(0.0, 5.0, 101))
    assert profile.npv.shape == (101,)
    for r in profile.irrs:
        assert abs(npv(r, [-1600.0, 10_000.0, -10_000.0])) < 1e-6


def test_irr_all_needs_two_cash_flows():
    """A single nonzero flow, after trailing zeros are dropped, has no rate that zeroes its NPV."""
    assert irr_all([5.0]).size == 0
    assert irr_all([-100.0, 0.0]).size == 0
    with pytest.raises(ValueError, match="non-empty"):
        irr_all([])
