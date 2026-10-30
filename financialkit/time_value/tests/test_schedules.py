"""Tests for coupon schedules, business-day adjustments and settlement lags."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL, DayCount
from financialkit.time_value import (
    BUSINESS_DAY_CONVENTIONS,
    STUBS,
    add_months,
    adjust_date,
    coupon_schedule,
    is_business_day,
    is_end_of_month,
    settlement_date,
)


def _iso(dates):
    return [str(d) for d in dates]


@pytest.mark.parametrize(
    "convention, expected",
    [
        ("unadjusted", "2023-09-30"),
        ("following", "2023-10-02"),
        ("modified following", "2023-09-29"),
        ("preceding", "2023-09-29"),
        ("modified preceding", "2023-09-29"),
    ],
)
def test_isda_business_day_conventions_at_a_month_end(convention, expected):
    """2006 ISDA Definitions, Sec. 4.12: Saturday 30 September 2023 rolls to
    Monday under following, but modified following stays in September."""
    assert str(adjust_date("2023-09-30", convention)) == expected


def test_modified_preceding_crosses_back_into_the_month():
    """Sunday 1 October 2023: preceding goes back to September, so modified
    preceding goes forward to Monday 2 October."""
    assert str(adjust_date("2023-10-01", "preceding")) == "2023-09-29"
    assert str(adjust_date("2023-10-01", "modified_preceding")) == "2023-10-02"


def test_holidays_are_user_supplied():
    """No calendar is bundled: Christmas is a business day until it is passed in."""
    assert is_business_day("2024-12-25")
    assert not is_business_day("2024-12-25", holidays=["2024-12-25"])
    assert str(adjust_date("2024-12-25", "following", holidays=["2024-12-25", "2024-12-26"])) == "2024-12-27"
    assert _iso(adjust_date(["2024-12-24", "2024-12-25"], "preceding", holidays=["2024-12-25"])) == ["2024-12-24", "2024-12-24"]


def test_settlement_lag_skips_weekends_and_holidays():
    """T + 2 from Friday 8 March 2024 is Tuesday; a Monday holiday pushes it to
    Wednesday; T + 0 from a Saturday settles on Monday."""
    assert str(settlement_date("2024-03-08")) == "2024-03-12"
    assert str(settlement_date("2024-03-08", 2, holidays=["2024-03-11"])) == "2024-03-13"
    assert str(settlement_date("2024-03-09", 0)) == "2024-03-11"
    assert str(settlement_date("2024-03-08", 1)) == "2024-03-11"


def test_unknown_convention_and_negative_lag_raise():
    with pytest.raises(ValueError, match="business-day convention"):
        adjust_date("2024-01-01", "nearest")
    with pytest.raises(ValueError, match="lag"):
        settlement_date("2024-01-01", -1)
    assert "modified following" in BUSINESS_DAY_CONVENTIONS


def test_add_months_clamps_and_end_of_month_rule():
    """31 August less six months is 28 February; with the end-of-month rule a
    roll from 28 February 2023 (a month end) stays on month ends."""
    assert _iso(add_months("2023-08-31", [-6, -3, 6])) == ["2023-02-28", "2023-05-31", "2024-02-29"]
    assert str(add_months("2023-02-28", 3)) == "2023-05-28"
    assert str(add_months("2023-02-28", 3, end_of_month=True)) == "2023-05-31"
    assert str(add_months("2023-02-15", 3, end_of_month=True)) == "2023-05-15"
    assert is_end_of_month(["2024-02-29", "2024-02-28", "2024-12-31"]).tolist() == [True, False, True]


def test_regular_schedule_is_the_same_for_every_stub():
    dates = ["2020-01-15", "2020-07-15", "2021-01-15", "2021-07-15", "2022-01-15"]
    for stub in STUBS:
        s = coupon_schedule("2020-01-15", "2022-01-15", 2, stub)
        assert _iso(s.unadjusted) == dates
        assert s.n_periods == 4
        np.testing.assert_allclose(s.accrual_fractions(), 0.5, rtol=DEFAULT_RTOL)


@pytest.mark.parametrize(
    "stub, dates",
    [
        ("short front", ["2020-02-01", "2020-07-15", "2021-01-15", "2021-07-15", "2022-01-15"]),
        ("long front", ["2020-02-01", "2021-01-15", "2021-07-15", "2022-01-15"]),
        ("short back", ["2020-02-01", "2020-08-01", "2021-02-01", "2021-08-01", "2022-01-15"]),
        ("long back", ["2020-02-01", "2020-08-01", "2021-02-01", "2022-01-15"]),
    ],
)
def test_stubs(stub, dates):
    """Front stubs roll back from maturity, back stubs forward from the start;
    a long stub absorbs the neighbouring regular period."""
    assert _iso(coupon_schedule("2020-02-01", "2022-01-15", 2, stub).unadjusted) == dates


def test_end_of_month_schedule():
    """ISDA 2006, Sec. 4.13: a quarterly roll from 28 February 2022 (a month
    end) lands on 30 November, 31 August and 31 May under the end-of-month
    rule, and on the 28th without it."""
    eom = coupon_schedule("2021-02-28", "2022-02-28", 4, end_of_month=True)
    assert _iso(eom.unadjusted) == ["2021-02-28", "2021-05-31", "2021-08-31", "2021-11-30", "2022-02-28"]
    plain = coupon_schedule("2021-02-28", "2022-02-28", 4)
    assert _iso(plain.unadjusted) == ["2021-02-28", "2021-05-28", "2021-08-28", "2021-11-28", "2022-02-28"]


def test_adjusted_dates_keep_the_unadjusted_roll():
    """Modified following moves 15 September 2024 (a Sunday) to Monday 16th,
    and the next roll is still 15 March, not 16 March."""
    s = coupon_schedule("2024-03-15", "2025-09-15", 2, convention="modified following", holidays=["2025-09-15"])
    assert _iso(s.unadjusted) == ["2024-03-15", "2024-09-15", "2025-03-15", "2025-09-15"]
    assert _iso(s.dates) == ["2024-03-15", "2024-09-16", "2025-03-17", "2025-09-16"]
    assert s.convention == "modified following"
    np.testing.assert_allclose(s.accrual_fractions(DayCount.ACT_360), np.array([185, 182, 183]) / 360, rtol=DEFAULT_RTOL)
    np.testing.assert_allclose(s.accrual_fractions(DayCount.ACT_360, adjusted=False), np.array([184, 181, 184]) / 360, rtol=DEFAULT_RTOL)


def test_isda_1998_icma_actual_actual_examples():
    """ISDA, *EMU and Market Conventions* (1998), Sec. 3, the ISMA column:
    Example 1, a regular half year 1 Nov 2003 - 1 May 2004, is 0.5;
    Example 2, a short first annual period 1 Feb - 1 Jul 1999, is 150/365 =
    0.41096, and the next full period is 1.0; Example 3, a long first
    semiannual period 15 Aug 2002 - 15 Jul 2003, is (153/184 + 181/181)/2 =
    0.91576."""
    np.testing.assert_allclose(coupon_schedule("2003-11-01", "2004-05-01", 2).accrual_fractions(), [0.5], rtol=DEFAULT_RTOL)
    short = coupon_schedule("1999-02-01", "2000-07-01", 1)
    np.testing.assert_allclose(short.accrual_fractions(), [150 / 365, 1.0], rtol=DEFAULT_RTOL)
    long = coupon_schedule("2002-08-15", "2004-01-15", 2, "long front")
    assert _iso(long.unadjusted) == ["2002-08-15", "2003-07-15", "2004-01-15"]
    np.testing.assert_allclose(long.accrual_fractions(), [(153 / 184 + 1.0) / 2, 0.5], rtol=DEFAULT_RTOL)
    assert round(float(long.accrual_fractions()[0]), 5) == 0.91576


def test_periods_between_on_30_360_and_outside_the_schedule():
    """On 30/360 a semiannual quasi-period is 180 days, so 1 March to 3 July
    is 122/180 of a period (Hull, 10th ed., Sec. 6.1)."""
    s = coupon_schedule("2017-09-01", "2020-09-01", 2)
    assert s.periods_between("2018-03-01", "2018-07-03", DayCount.THIRTY_360_US) == pytest.approx(122 / 180, rel=DEFAULT_RTOL)
    assert s.periods_between("2018-03-01", "2018-07-03") == pytest.approx(124 / 184, rel=DEFAULT_RTOL)
    with pytest.raises(ValueError, match="quasi-coupon"):
        s.periods_between("2010-01-01", "2018-07-03")


def test_short_period_shorter_than_one_coupon():
    """A deal shorter than one period is a single stub, long or short."""
    for stub in STUBS:
        assert _iso(coupon_schedule("2024-01-10", "2024-05-15", 2, stub).unadjusted) == ["2024-01-10", "2024-05-15"]


def test_schedule_validation():
    with pytest.raises(ValueError, match="end must be after start"):
        coupon_schedule("2024-01-01", "2024-01-01")
    with pytest.raises(ValueError, match="freq"):
        coupon_schedule("2024-01-01", "2025-01-01", 5)
    with pytest.raises(ValueError, match="stub"):
        coupon_schedule("2024-01-01", "2025-01-01", 2, "middle")
