"""Tests for bonds and swaps on dated schedules: odd coupons, accrual and business-day payment dates."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL, DayCount
from financialkit.fixed_income import (
    FlatCurve,
    ZeroCurve,
    bond_accrued_interest,
    bond_cash_flows,
    bond_price,
    clean_price,
    currency_swap_value,
    interest_rate_swap_value,
    swap_annuity,
    swap_rate,
    yield_to_maturity,
)
from financialkit.time_value import coupon_schedule, year_fraction

# A 7.85% semiannual bond dated 15 October 2008 with a short first coupon on
# 1 March 2009, maturing 1 March 2021, settling 11 November 2008.
SHORT_FIRST = coupon_schedule("2008-10-15", "2021-03-01", 2)
SETTLE = "2008-11-11"


def test_short_first_coupon_prices_by_the_sia_formula():
    """SIA, *Standard Securities Calculation Methods* (Mayle, 3rd ed., 1993),
    odd short first coupon: with E = 181 days in the quasi-coupon period
    1 Sep 2008 - 1 Mar 2009, DFC = 137 days from the dated date to the first
    coupon, A = 27 accrued days and DSC = 110 days to the first coupon,

    P = 100 c/2 DFC/E v^(DSC/E) + sum_k 100 c/2 v^(k-1+DSC/E) + 100 v^(N-1+DSC/E) - 100 c/2 A/E

    with v = 1/(1 + y/2). Excel's ``ODDFPRICE`` documentation prices this
    bond at a 6.25% yield (actual/actual) at 113.597717."""
    c, y, E, DFC, A, DSC = 0.0785, 0.0625, 181, 137, 27, 110
    N = 25
    v = 1 / (1 + y / 2)
    k = np.arange(2, N + 1)
    sia = 100 * c / 2 * DFC / E * v ** (DSC / E) + np.sum(100 * c / 2 * v ** (k - 1 + DSC / E)) + 100 * v ** (N - 1 + DSC / E) - 100 * c / 2 * A / E
    price = clean_price(100.0, c, None, y, compounding=2, schedule=SHORT_FIRST, settlement=SETTLE)
    assert price == pytest.approx(sia, rel=DEFAULT_RTOL)
    assert price == pytest.approx(113.597717, abs=5e-7)
    assert bond_accrued_interest(100.0, c, schedule=SHORT_FIRST, settlement=SETTLE) == pytest.approx(100 * c / 2 * A / E, rel=DEFAULT_RTOL)
    cf = bond_cash_flows(100.0, c, schedule=SHORT_FIRST, settlement=SETTLE)
    assert len(cf.times) == N
    assert cf.amounts[0] == pytest.approx(100 * c / 2 * DFC / E, rel=DEFAULT_RTOL)
    np.testing.assert_allclose(cf.times, (np.arange(N) + DSC / E) / 2, rtol=DEFAULT_RTOL)


def test_short_first_coupon_yield_on_30_360():
    """Excel's ``ODDFYIELD`` documentation: the same dates with a 5.75% coupon
    priced at 84.50 on a 30/360 basis yield 7.72%. On 30/360 the first
    coupon is 136/180 of a regular one (15 Oct to 1 Mar is 136 days)."""
    cf = bond_cash_flows(100.0, 0.0575, schedule=SHORT_FIRST, settlement=SETTLE, day_count=DayCount.THIRTY_360_US)
    ai = bond_accrued_interest(100.0, 0.0575, schedule=SHORT_FIRST, settlement=SETTLE, day_count="30/360 US")
    assert cf.amounts[0] == pytest.approx(100 * 0.0575 / 2 * 136 / 180, rel=DEFAULT_RTOL)
    assert ai == pytest.approx(100 * 0.0575 / 2 * 26 / 180, rel=DEFAULT_RTOL)
    assert round(yield_to_maturity(84.50 + ai, cf, compounding=2), 4) == 0.0772


def test_long_first_coupon_by_the_sia_formula():
    """SIA odd long first coupon: dated 15 Aug 2002, first coupon 15 Jul 2003,
    so the coupon spans two quasi-coupon periods (184 and 181 days) and pays
    c/2 (153/184 + 181/181). Settling 1 Oct 2002, the buyer owes 47/184 of a
    coupon, and the first coupon is 106/184 + 1 periods away."""
    s = coupon_schedule("2002-08-15", "2005-01-15", 2, "long front")
    c, y = 0.06, 0.05
    v = 1 / (1 + y / 2)
    first = 100 * c / 2 * (153 / 184 + 1.0)
    tau = 106 / 184 + 1 + np.arange(4)
    flows = np.array([first, 3.0, 3.0, 103.0])
    dirty = bond_price(100.0, c, None, y, compounding=2, schedule=s, settlement="2002-10-01")
    assert dirty == pytest.approx(np.sum(flows * v**tau), rel=DEFAULT_RTOL)
    assert bond_accrued_interest(100.0, c, schedule=s, settlement="2002-10-01") == pytest.approx(3.0 * 47 / 184, rel=DEFAULT_RTOL)


def test_hull_treasury_and_corporate_accrual():
    """Hull (10th ed.), Sec. 6.1: a 10% bond paying 1 March and 1 September
    settling 3 July 2018 has accrued 124/184 * 5 = 3.3696 on actual/actual
    (Treasury) and 122/180 * 5 = 3.3889 on 30/360 (corporate)."""
    s = coupon_schedule("2017-09-01", "2020-09-01", 2)
    assert bond_accrued_interest(100.0, 0.10, schedule=s, settlement="2018-07-03") == pytest.approx(124 / 184 * 5, rel=DEFAULT_RTOL)
    assert bond_accrued_interest(100.0, 0.10, schedule=s, settlement="2018-07-03", day_count="30/360 US") == pytest.approx(122 / 180 * 5, rel=DEFAULT_RTOL)


def test_regular_dated_bond_matches_the_year_fraction_bond():
    """On a coupon date of a regular schedule the dated bond is the undated
    one, so it prices at par at its coupon rate and has no accrued interest."""
    s = coupon_schedule("2020-01-15", "2030-01-15", 2)
    dated = bond_price(100.0, 0.05, None, [0.04, 0.05, 0.06], compounding=2, schedule=s, settlement="2022-01-15")
    np.testing.assert_allclose(dated, bond_price(100.0, 0.05, 8.0, [0.04, 0.05, 0.06], compounding=2), rtol=DEFAULT_RTOL)
    assert bond_accrued_interest(100.0, 0.05, schedule=s, settlement="2022-01-15") == 0.0


def test_dated_zero_coupon_bond():
    cf = bond_cash_flows(100.0, 0.0, schedule=SHORT_FIRST, settlement=SETTLE)
    assert cf.amounts.tolist() == [100.0]
    assert cf.times[0] == pytest.approx((24 + 110 / 181) / 2, rel=DEFAULT_RTOL)


def test_dated_bond_validation():
    with pytest.raises(ValueError, match="either maturity or schedule"):
        bond_cash_flows(100.0, 0.05)
    with pytest.raises(ValueError, match="either maturity or schedule"):
        bond_cash_flows(100.0, 0.05, 2.0, schedule=SHORT_FIRST, settlement=SETTLE)
    with pytest.raises(ValueError, match="settlement is required"):
        bond_cash_flows(100.0, 0.05, schedule=SHORT_FIRST)
    for bad in ("2008-10-01", "2021-03-01"):
        with pytest.raises(ValueError, match="settlement must be"):
            bond_accrued_interest(100.0, 0.05, schedule=SHORT_FIRST, settlement=bad)


# A five-year semiannual swap from Friday 15 March 2024, modified following,
# with a holiday on Tuesday 15 September 2026.
SWAP = coupon_schedule("2024-03-15", "2029-03-15", 2, convention="modified following", holidays=["2026-09-15"])
CURVE = ZeroCurve([0.5, 1.0, 2.0, 5.0, 10.0], [0.03, 0.032, 0.035, 0.037, 0.04])


def test_dated_swap_pays_on_business_days_and_prices_at_par():
    """The par rate makes a new swap worth zero, and the annuity is the sum of
    ACT/360 accruals between adjusted dates times their discount factors."""
    assert str(SWAP.dates[1]) == "2024-09-16" and str(SWAP.dates[5]) == "2026-09-16"
    alpha = year_fraction(SWAP.dates[:-1], SWAP.dates[1:], DayCount.ACT_360)
    P = CURVE.discount(year_fraction(SWAP.dates[0], SWAP.dates[1:], DayCount.ACT_365F))
    assert swap_annuity(CURVE, schedule=SWAP) == pytest.approx(np.sum(alpha * P), rel=DEFAULT_RTOL)
    K = swap_rate(CURVE, schedule=SWAP)
    assert K == pytest.approx((1 - P[-1]) / np.sum(alpha * P), rel=DEFAULT_RTOL)
    for pay_fixed in (True, False):
        assert interest_rate_swap_value(CURVE, 100.0, K, schedule=SWAP, pay_fixed=pay_fixed).value == pytest.approx(0.0, abs=1e-10)


def test_forward_starting_dated_swap():
    """Valued before its start, the floating leg is the notional discounted
    from the start date, and the par rate is the forward swap rate."""
    v = "2023-09-15"
    K = swap_rate(CURVE, schedule=SWAP, valuation_date=v)
    val = interest_rate_swap_value(CURVE, 100.0, K, schedule=SWAP, valuation_date=v)
    assert val.value == pytest.approx(0.0, abs=1e-10)
    assert val.receive_leg == pytest.approx(100 * float(CURVE.discount(year_fraction(v, "2024-03-15", "ACT/365F"))), rel=DEFAULT_RTOL)


def test_seasoned_dated_swap():
    """Between resets the floating leg is the notional plus the fixed coupon,
    discounted from the next payment date; on a reset date it is the notional."""
    v = "2025-01-10"
    alpha = year_fraction("2024-09-16", "2025-03-17", "ACT/360")
    P_next = float(CURVE.discount(year_fraction(v, "2025-03-17", "ACT/365F")))
    val = interest_rate_swap_value(CURVE, 100.0, 0.035, schedule=SWAP, valuation_date=v, floating_rate=0.03)
    assert val.receive_leg == pytest.approx(100 * (1 + 0.03 * alpha) * P_next, rel=DEFAULT_RTOL)
    on_reset = interest_rate_swap_value(CURVE, 100.0, 0.035, schedule=SWAP, valuation_date="2025-03-17", pay_fixed=False)
    assert on_reset.pay_leg == 100.0
    with pytest.raises(ValueError, match="floating_rate is required"):
        interest_rate_swap_value(CURVE, 100.0, 0.035, schedule=SWAP, valuation_date=v)
    with pytest.raises(ValueError, match="on or before"):
        swap_rate(CURVE, schedule=SWAP, valuation_date=v)
    with pytest.raises(ValueError, match="before the schedule's last payment"):
        swap_annuity(CURVE, schedule=SWAP, valuation_date="2029-03-15")
    with pytest.raises(ValueError, match="either maturity or schedule"):
        swap_rate(CURVE)


def test_dated_currency_swap_at_par_rates_is_worth_zero():
    """Each fixed leg with principal is worth its notional at its par rate, so
    exchanging par legs of equal value at spot is worth nothing."""
    s = coupon_schedule("2024-01-02", "2027-01-04", 1, convention="following")
    dom, fgn = FlatCurve(0.09), FlatCurve(0.04)
    Kd, Kf = swap_rate(dom, schedule=s), swap_rate(fgn, schedule=s)
    v = currency_swap_value(dom, fgn, 1 / 110, 10.0, 1100.0, Kd, Kf, schedule=s)
    assert v.receive_leg == pytest.approx(10.0, rel=DEFAULT_RTOL)
    assert v.value == pytest.approx(0.0, abs=1e-10)
    flipped = currency_swap_value(dom, fgn, 1 / 110, 10.0, 1100.0, Kd + 0.01, Kf, schedule=s, receive_foreign=False)
    assert flipped.value > 0
