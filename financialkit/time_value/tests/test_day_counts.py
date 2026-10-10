"""Tests for day counts, year fractions and accrued interest."""

import datetime

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL, DayCount
from financialkit.time_value import accrued_interest, day_count, days_in_year, is_leap_year, year_fraction


def test_hull_section_6_1_day_counts_and_accrued_interest():
    """Hull (10th ed.), Sec. 6.1: from March 1 to July 3, 2018 there are 124
    actual days and 122 days on 30/360; a 10% bond with coupons on March 1 and
    September 1 has accrued 122/180 * 5 = 3.3889 under 30/360."""
    assert day_count("2018-03-01", "2018-07-03", DayCount.ACT_360) == 124
    assert day_count("2018-03-01", "2018-07-03", DayCount.THIRTY_360_US) == 122
    assert accrued_interest(100.0, 0.10, "2018-03-01", "2018-07-03") == pytest.approx(122 / 180 * 5, rel=DEFAULT_RTOL)


def test_isda_1998_actual_actual_example():
    """ISDA, *EMU and Market Conventions* (1998), Sec. 3, Example 1:
    1 Nov 2003 to 1 May 2004 is 61/365 + 121/366 = 0.49772438 under ACT/ACT ISDA,
    against 182/365 = 0.49863014 under ACT/365F."""
    assert year_fraction("2003-11-01", "2004-05-01", DayCount.ACT_ACT_ISDA) == pytest.approx(61 / 365 + 121 / 366, rel=DEFAULT_RTOL)
    assert year_fraction("2003-11-01", "2004-05-01", DayCount.ACT_365F) == pytest.approx(182 / 365, rel=DEFAULT_RTOL)


@pytest.mark.parametrize(
    "start, end, us, euro",
    [
        ("2023-01-31", "2023-03-31", 60, 60),  # D1 = 31 -> 30, then D2 = 31 -> 30
        ("2023-01-30", "2023-03-31", 60, 60),
        ("2023-01-15", "2023-03-31", 76, 75),  # US keeps D2 = 31 when D1 < 30
        ("2023-02-28", "2023-08-31", 180, 182),  # US end-of-February rule
        ("2024-02-29", "2025-02-28", 360, 359),  # both end of February
        ("2024-02-29", "2024-03-31", 30, 31),
    ],
)
def test_thirty_360_us_vs_european(start, end, us, euro):
    """SIA *Standard Securities Calculation Methods* (1993) 30/360 rules vs 30E/360."""
    assert day_count(start, end, "30/360 US") == us
    assert day_count(start, end, "30E/360") == euro


@pytest.mark.parametrize("convention", list(DayCount))
def test_every_convention_vectorised_and_about_one_per_year(convention):
    starts = np.array(["2020-01-15", "2021-06-30", "2023-12-31"], dtype="datetime64[D]")
    ends = starts + np.array([365, 366, 365])
    yf = year_fraction(starts, ends, convention)
    assert yf.shape == (3,)
    np.testing.assert_allclose(yf, 1.0, atol=0.02)
    assert year_fraction("2024-01-01", "2024-01-01", convention) == 0.0


def test_act_act_isda_whole_years_and_accepts_dates():
    assert year_fraction(datetime.date(2020, 1, 1), datetime.date(2025, 1, 1), "ACT/ACT ISDA") == pytest.approx(5.0, rel=DEFAULT_RTOL)
    assert year_fraction("2024-01-01", "2024-07-01", "ACT/360") == pytest.approx(182 / 360, rel=DEFAULT_RTOL)


def test_leap_years():
    np.testing.assert_array_equal(is_leap_year([1900, 2000, 2023, 2024]), [False, True, False, True])
    np.testing.assert_array_equal(days_in_year([2023, 2024]), [365, 366])


def test_unknown_convention_raises():
    with pytest.raises(ValueError):
        year_fraction("2024-01-01", "2024-02-01", "ACT/999")
