"""Day counts, year fractions and accrued interest.

A day-count convention turns a pair of dates into a year fraction: the
numerator counts days (actual or on a 30-day-month calendar), the
denominator is a notional year. The 30/360 US rules with their
end-of-February adjustments are those of the Securities Industry
Association's *Standard Securities Calculation Methods* (Mayle, 3rd ed.,
1993); 30E/360 and actual/actual ISDA are as defined in the 2006 ISDA
Definitions, Sec. 4.16, and ISDA's 1998 memo *EMU and Market Conventions:
Recent Developments*, which set out the actual/actual variants side by
side. Accrued interest is the coupon earned since the last coupon date,
``face * coupon_rate * year_fraction(last_coupon, settlement)``.

Dates are anything :func:`numpy.asarray` turns into ``datetime64[D]``
(:class:`datetime.date`, ``numpy.datetime64`` or ISO strings); every
function broadcasts over arrays of dates.
"""

from __future__ import annotations

import numpy as np

from financialkit.constants import DayCount
from financialkit.time_value.core.base import _finalize
from financialkit.time_value.utils.dates import as_dates, date_parts, days_in_year, is_end_of_february

__all__ = ["day_count", "year_fraction", "accrued_interest"]


def _thirty_360_days(start, end, european: bool) -> np.ndarray:
    y1, m1, d1 = date_parts(start)
    y2, m2, d2 = date_parts(end)
    d1, d2 = (a.copy() for a in np.broadcast_arrays(d1, d2))
    if european:
        d1 = np.minimum(d1, 30)
        d2 = np.minimum(d2, 30)
    else:
        feb1 = np.broadcast_to(is_end_of_february(start), d1.shape)
        feb2 = np.broadcast_to(is_end_of_february(end), d2.shape)
        d2 = np.where(feb1 & feb2, 30, d2)
        d1 = np.where(feb1, 30, d1)
        d2 = np.where((d2 == 31) & (d1 >= 30), 30, d2)
        d1 = np.where(d1 == 31, 30, d1)
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)


def day_count(start, end, convention=DayCount.ACT_365F):
    """Days between two dates under a day-count convention.

    Parameters
    ----------
    start, end : date-like or array-like of date-likes
    convention : DayCount or str
        One of :class:`~financialkit.constants.DayCount`; the ``ACT``
        conventions count actual days.

    Returns
    -------
    int or ndarray of int

    Examples
    --------
    Hull (10th ed.), Sec. 6.1: March 1 to July 3, 2018.

    >>> day_count("2018-03-01", "2018-07-03", "30/360 US"), day_count("2018-03-01", "2018-07-03", "ACT/360")
    (122, 124)
    """
    convention = DayCount(convention)
    if convention in (DayCount.THIRTY_360_US, DayCount.THIRTY_E_360):
        days = _thirty_360_days(start, end, european=convention is DayCount.THIRTY_E_360)
    else:
        days = (as_dates(end) - as_dates(start)).astype(np.int64)
    days = np.asarray(days)
    return int(days) if days.ndim == 0 else days


def year_fraction(start, end, convention=DayCount.ACT_365F):
    """Year fraction between two dates.

    Parameters
    ----------
    start, end : date-like or array-like of date-likes
    convention : DayCount or str

    Returns
    -------
    float or ndarray

    Examples
    --------
    ISDA (1998), Sec. 3, Example 1: November 1, 2003 to May 1, 2004 is
    61/365 + 121/366 under actual/actual ISDA.

    >>> round(year_fraction("2003-11-01", "2004-05-01", "ACT/ACT ISDA"), 5)
    0.49772
    """
    convention = DayCount(convention)
    if convention is DayCount.ACT_360:
        return _finalize(np.asarray(day_count(start, end, convention)) / 360.0)
    if convention is DayCount.ACT_365F:
        return _finalize(np.asarray(day_count(start, end, convention)) / 365.0)
    if convention is DayCount.ACT_ACT_ISDA:
        s, e = as_dates(start), as_dates(end)
        y1, y2 = date_parts(s)[0], date_parts(e)[0]
        jan1 = lambda y: (np.asarray(y) - 1970).astype("datetime64[Y]").astype("datetime64[D]")
        into1 = (s - jan1(y1)).astype(np.int64) / days_in_year(y1)
        into2 = (e - jan1(y2)).astype(np.int64) / days_in_year(y2)
        return _finalize((y2 - y1) + into2 - into1)
    return _finalize(np.asarray(day_count(start, end, convention)) / 360.0)


def accrued_interest(face, coupon_rate, last_coupon, settlement, convention=DayCount.THIRTY_360_US):
    """Coupon interest accrued from the last coupon date to settlement.

    This is the calendar version: the accrual period runs between two dates
    and is measured with a day-count `convention`. For a bond described by
    its time to maturity in years and a regular coupon frequency, use
    :func:`financialkit.fixed_income.bond_accrued_interest`.

    Parameters
    ----------
    face : float or array-like
    coupon_rate : float or array-like
        Annual coupon rate.
    last_coupon, settlement : date-like or array-like of date-likes
    convention : DayCount or str

    Returns
    -------
    float or ndarray

    Examples
    --------
    Hull (10th ed.), Sec. 6.1: a 10% corporate bond, coupons on March 1
    and September 1, settling July 3, 2018 (30/360).

    >>> round(accrued_interest(100.0, 0.10, "2018-03-01", "2018-07-03"), 4)
    3.3889
    """
    yf = np.asarray(year_fraction(last_coupon, settlement, convention))
    return _finalize(np.asarray(face, dtype=np.float64) * np.asarray(coupon_rate, dtype=np.float64) * yf)
