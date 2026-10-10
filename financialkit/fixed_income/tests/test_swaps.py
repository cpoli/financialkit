"""Tests for interest-rate and currency swap valuation."""

import numpy as np
import pytest

from financialkit.fixed_income import (
    FlatCurve,
    ZeroCurve,
    bond_cash_flows,
    currency_swap_value,
    interest_rate_swap_value,
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
