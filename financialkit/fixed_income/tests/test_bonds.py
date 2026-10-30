"""Tests for bond pricing, yield to maturity, accrued interest and STRIPS."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.fixed_income import (
    CashFlows,
    FlatCurve,
    ZeroCurve,
    bond_accrued_interest,
    bond_cash_flows,
    bond_price,
    clean_price,
    coupon_times,
    price_from_yield,
    strips_curve,
    strips_decomposition,
    strips_value,
    yield_to_maturity,
    zero_coupon_bond_price,
)


def test_hull_two_year_bond_price_and_yield():
    """Hull, 9th ed., Sec. 4.4: 2-year 6% semiannual bond, yield 6.76% (cont.) <-> price 98.39."""
    assert bond_price(100.0, 0.06, 2.0, 0.0676) == pytest.approx(98.39, abs=5e-3)
    assert yield_to_maturity(98.39, bond_cash_flows(100.0, 0.06, 2.0)) == pytest.approx(0.0676, abs=5e-5)


@pytest.mark.parametrize("compounding", [None, 1, 2, 12])
def test_yield_to_maturity_round_trip(compounding):
    cf = bond_cash_flows(100.0, 0.045, 7.3, freq=2)
    for y in (-0.005, 0.0, 0.03, 0.12):
        price = price_from_yield(cf, y, compounding)
        assert yield_to_maturity(price, cf, compounding) == pytest.approx(y, abs=1e-9)


def test_bond_at_par_when_yield_equals_coupon_with_matching_compounding():
    for freq in (1, 2, 4):
        assert bond_price(100.0, 0.07, 10.0, 0.07, freq=freq, compounding=freq) == pytest.approx(100.0, rel=DEFAULT_RTOL)


def test_price_from_yield_broadcasts():
    y = np.array([0.01, 0.05, 0.09])
    prices = bond_price(100.0, 0.05, 5.0, y)
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)


def test_short_first_period_and_accrued_interest():
    cf = bond_cash_flows(100.0, 0.08, 1.375, freq=2)
    np.testing.assert_allclose(cf.times, [0.375, 0.875, 1.375])
    assert bond_accrued_interest(100.0, 0.08, 1.375) == pytest.approx(1.0)
    assert bond_accrued_interest(100.0, 0.08, 1.5) == pytest.approx(0.0, abs=1e-12)


def test_clean_price_is_continuous_across_coupon_date():
    """The dirty price drops by the coupon on a payment date; the clean price does not."""
    y = 0.05
    before, after = 2.0 + 1e-7, 2.0
    dirty_jump = bond_price(100.0, 0.06, before, y) - bond_price(100.0, 0.06, after, y)
    clean_jump = clean_price(100.0, 0.06, before, y) - clean_price(100.0, 0.06, after, y)
    assert dirty_jump == pytest.approx(3.0, abs=1e-4)
    assert clean_jump == pytest.approx(0.0, abs=1e-4)


def test_zero_coupon_bond_price_hull_table_4_3():
    """Hull, 9th ed., Table 4.3: 1-year zero priced 90 has a 10.536% continuous zero rate."""
    assert zero_coupon_bond_price(1.0, 0.10536) == pytest.approx(90.0, abs=1e-3)
    assert yield_to_maturity(90.0, CashFlows([1.0], [100.0])) == pytest.approx(-np.log(0.9), abs=1e-10)


def test_bond_equals_its_strips_on_any_curve():
    curve = ZeroCurve([0.5, 2.0, 5.0], [0.02, 0.035, 0.045])
    cf = bond_cash_flows(100.0, 0.05, 5.0)
    strips = strips_decomposition(cf)
    assert len(strips) == cf.times.size
    assert sum(s.present_value(curve) for s in strips) == pytest.approx(cf.present_value(curve), rel=DEFAULT_RTOL)
    quotes = 100.0 * curve.discount(cf.times)
    assert strips_value(cf, quotes) == pytest.approx(cf.present_value(curve), rel=DEFAULT_RTOL)
    assert strips_value(cf, curve) == pytest.approx(cf.present_value(curve), rel=DEFAULT_RTOL)


def test_strips_curve_inverts_strip_prices():
    curve = FlatCurve(0.04)
    t = np.array([0.5, 1.0, 3.0])
    implied = strips_curve(t, 100.0 * curve.discount(t))
    np.testing.assert_allclose(implied.rates, 0.04, rtol=DEFAULT_RTOL)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        yield_to_maturity(-1.0, bond_cash_flows(100.0, 0.05, 2.0))
    with pytest.raises(ValueError):
        strips_value(bond_cash_flows(100.0, 0.05, 2.0), [99.0])


def test_yield_to_maturity_expands_its_bracket_for_extreme_prices():
    """A 2-year zero worth 1000 or 1 per 100 face needs yields far outside the initial [-1, 1] bracket."""
    zero = CashFlows([2.0], [100.0])
    assert yield_to_maturity(1000.0, zero) == pytest.approx(-np.log(10.0) / 2, abs=1e-10)
    assert yield_to_maturity(1.0, zero) == pytest.approx(np.log(100.0) / 2, abs=1e-10)


def test_yield_to_maturity_rejects_prices_no_yield_can_reach():
    """A cash flow due now is worth its face at every yield, so any other price has no yield."""
    due_now = CashFlows([0.0], [100.0])
    with pytest.raises(ValueError, match="above any attainable"):
        yield_to_maturity(150.0, due_now)
    with pytest.raises(ValueError, match="below any attainable"):
        yield_to_maturity(50.0, due_now)
    with pytest.raises(ValueError, match="nonnegative cash flows"):
        yield_to_maturity(100.0, CashFlows([1.0, 2.0], [110.0, -5.0]))


def test_coupon_times_and_cash_flows_reject_invalid_inputs():
    with pytest.raises(ValueError, match="maturity must be positive"):
        coupon_times(0.0, 2)
    with pytest.raises(ValueError, match="freq must be a positive integer"):
        coupon_times(1.0, 0)
    with pytest.raises(ValueError, match="same shape"):
        CashFlows([1.0, 2.0], [5.0])
