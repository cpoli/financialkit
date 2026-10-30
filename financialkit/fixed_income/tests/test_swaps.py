"""Tests for interest-rate and currency swap valuation."""

import numpy as np
import pytest

from financialkit.fixed_income import (
    FlatCurve,
    ZeroCurve,
    bond_cash_flows,
    compounded_in_arrears,
    currency_swap_value,
    interest_rate_swap_value,
    overnight_forwards,
    swap_annuity,
    swap_rate,
)


def test_hull_example_7_1_interest_rate_swap():
    """Hull, 9th ed., Example 7.1: B_fix = 98.238, B_fl = 102.505, swap worth -4.267 to the fixed receiver."""
    curve = ZeroCurve([0.25, 0.75, 1.25], [0.10, 0.105, 0.11])
    v = interest_rate_swap_value(curve, 100.0, 0.08, 1.25, pay_fixed=False, floating_rate=0.102)
    assert v.receive_leg == pytest.approx(98.238, abs=1e-3)
    assert v.pay_leg == pytest.approx(102.505, abs=1e-3)
    assert v.value == pytest.approx(-4.267, abs=1e-3)
    payer = interest_rate_swap_value(curve, 100.0, 0.08, 1.25, pay_fixed=True, floating_rate=0.102)
    assert payer.value == pytest.approx(-v.value)


def test_hull_example_7_3_currency_swap():
    """Hull, 9th ed., Example 7.3: B_D = 9.6439 $m, B_F = 1230.55 yen m, swap worth 1.5430 $m."""
    v = currency_swap_value(0.09, 0.04, 1 / 110, 10.0, 1200.0, 0.08, 0.05, 3.0)
    assert v.pay_leg == pytest.approx(9.6439, abs=1e-4)
    assert v.receive_leg * 110 == pytest.approx(1230.55, abs=1e-2)
    assert v.value == pytest.approx(1.5430, abs=1e-3)


def test_new_swap_at_par_rate_is_worth_zero():
    curve = ZeroCurve([0.5, 2, 5, 10], [0.02, 0.028, 0.035, 0.04])
    for T in (2.0, 5.0, 10.0):
        K = swap_rate(curve, T)
        assert interest_rate_swap_value(curve, 1e6, K, T).value == pytest.approx(0.0, abs=1e-6)
        assert K == pytest.approx(curve.par_yield(T), rel=1e-14)


def test_swap_value_is_annuity_times_rate_difference():
    curve = FlatCurve(0.03)
    K, T = 0.025, 7.0
    v = interest_rate_swap_value(curve, 100.0, K, T, freq=4)
    assert v.value == pytest.approx(100.0 * (swap_rate(curve, T, freq=4) - K) * swap_annuity(curve, T, freq=4), rel=1e-12)


def test_floating_rate_required_between_resets():
    with pytest.raises(ValueError):
        interest_rate_swap_value(0.03, 100.0, 0.03, 1.25)


def test_currency_swap_with_equal_curves_and_rates_is_principal_conversion():
    v = currency_swap_value(0.03, 0.03, 1.2, 120.0, 100.0, 0.04, 0.04, 5.0)
    assert v.value == pytest.approx(0.0, abs=1e-10)
    flipped = currency_swap_value(0.03, 0.01, 1.2, 120.0, 100.0, 0.04, 0.04, 5.0, receive_foreign=False)
    assert flipped.value < 0
    assert flipped.pay_leg == pytest.approx(1.2 * bond_cash_flows(100.0, 0.04, 5.0, 1).present_value(0.01))
    assert np.isfinite(flipped.receive_leg)


DAYS = np.arange("2024-01-01", "2024-07-01", dtype="datetime64[D]")
BUSINESS = DAYS[np.is_busday(DAYS)]


def test_compounded_in_arrears_constant_rate():
    r = compounded_in_arrears(BUSINESS, np.full(BUSINESS.size, 0.05), "2024-01-02", "2024-04-02")
    expected = (np.prod(1 + 0.05 * r.weights / 360) - 1) * 360 / 91
    assert r.rate == pytest.approx(expected, rel=1e-14)
    assert r.weights.sum() == 91 and r.accrual == pytest.approx(91 / 360)
    assert r.cumulative[-1] == pytest.approx(r.rate, rel=1e-14)
    assert r.rate > 0.05  # compounding


def test_compounded_rate_of_curve_forwards_is_simple_forward():
    """ARRC/ISDA: the expected compounded coupon is the OIS forward, so a SOFR leg is worth P(t0) - P(tn)."""
    curve = ZeroCurve([0.25, 1.0, 2.0], [0.050, 0.046, 0.042])
    holidays = ["2024-01-15", "2024-02-19"]
    dates, rates = overnight_forwards(curve, "2024-04-02", "2024-07-02", valuation_date="2024-01-02", holidays=holidays)
    r = compounded_in_arrears(dates, rates, "2024-04-02", "2024-07-02", holidays=holidays)
    P = curve.discount(np.array([91, 182]) / 365)
    assert r.rate == pytest.approx((P[0] / P[1] - 1) / r.accrual, rel=1e-12)


def test_arrc_conventions():
    rng = np.random.default_rng(0)
    fixings = 0.053 + 0.0005 * rng.standard_normal(BUSINESS.size)
    base = compounded_in_arrears(BUSINESS, fixings, "2024-02-01", "2024-05-01")
    look = compounded_in_arrears(BUSINESS, fixings, "2024-02-01", "2024-05-01", lookback=2)
    shift = compounded_in_arrears(BUSINESS, fixings, "2024-02-01", "2024-05-01", lookback=2, observation_shift=True)
    lock = compounded_in_arrears(BUSINESS, fixings, "2024-02-01", "2024-05-01", lockout=2)
    assert look.weights.tolist() == base.weights.tolist()
    assert look.fixing_dates[0] == np.datetime64("2024-01-30")
    assert shift.dates[0] == np.datetime64("2024-01-30") and shift.weights.sum() == (np.datetime64("2024-04-29") - np.datetime64("2024-01-30")).astype(int)
    assert lock.rates[-1] == lock.rates[-2] == lock.rates[-3]
    assert lock.fixing_dates[-1] == base.dates[-3]
    for other in (look, shift, lock):
        assert abs(other.rate - base.rate) < 5e-4  # conventions move the coupon by basis points
    assert compounded_in_arrears(BUSINESS, fixings, "2024-02-01", "2024-05-01", lookback=0, lockout=0).rate == base.rate


def test_compounded_in_arrears_validation():
    with pytest.raises(ValueError, match="business days"):
        compounded_in_arrears(BUSINESS, np.zeros(BUSINESS.size), "2024-01-06", "2024-02-01")
    with pytest.raises(ValueError, match="no fixing"):
        compounded_in_arrears(BUSINESS[5:], np.zeros(BUSINESS.size - 5), "2024-01-02", "2024-02-01")
    with pytest.raises(ValueError):
        compounded_in_arrears(BUSINESS, np.zeros(BUSINESS.size), "2024-01-02", "2024-01-03", lockout=1)
    with pytest.raises(ValueError):
        compounded_in_arrears(BUSINESS, np.zeros(3), "2024-01-02", "2024-02-01")
    with pytest.raises(ValueError):
        compounded_in_arrears(BUSINESS, np.zeros(BUSINESS.size), "2024-02-01", "2024-01-02")
    with pytest.raises(ValueError):
        compounded_in_arrears(BUSINESS, np.zeros(BUSINESS.size), "2024-01-02", "2024-02-01", lookback=-1)


def test_overnight_forwards_needs_a_business_day():
    # Saturday to Monday holds no business day
    with pytest.raises(ValueError, match="no business day"):
        overnight_forwards(0.04, "2024-01-06", "2024-01-08")
