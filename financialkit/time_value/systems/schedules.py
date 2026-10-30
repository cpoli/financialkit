"""Coupon schedules, business-day adjustments and settlement lags.

A schedule rolls regular dates every ``12 / freq`` months from one end of
the deal. Rolling back from the end puts any irregular period (a *stub*)
at the front, and rolling forward from the start puts it at the back. A
short stub is kept on its own, and a long stub is merged into the regular
period next to it. Under the end-of-month rule, a roll from the last day
of a month stays on month ends. A date that falls on a weekend or a
holiday is then moved by a business-day convention: following, modified
following (following unless that crosses into the next month, then
preceding) or preceding. The conventions and the end-of-month rule are
those of the 2006 ISDA Definitions, Secs. 4.12-4.13. Odd first and last
coupons are measured against quasi-coupon periods, as in the Securities
Industry Association's *Standard Securities Calculation Methods* (Mayle,
3rd ed., 1993).

Business days are Monday to Friday less a user-supplied array of
holidays; no holiday calendar is bundled. The date arithmetic is
:func:`numpy.busday_offset` and :func:`numpy.is_busday`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from financialkit.time_value.core.base import Schedule
from financialkit.time_value.utils.dates import add_months, as_dates

__all__ = ["BUSINESS_DAY_CONVENTIONS", "STUBS", "is_business_day", "adjust_date", "settlement_date", "coupon_schedule"]

#: Business-day conventions and the :func:`numpy.busday_offset` roll each one maps to.
BUSINESS_DAY_CONVENTIONS: dict[str, Any] = {
    "unadjusted": None,
    "following": "following",
    "modified following": "modifiedfollowing",
    "preceding": "preceding",
    "modified preceding": "modifiedpreceding",
}

#: Stub placements accepted by :func:`coupon_schedule`.
STUBS = ("short front", "long front", "short back", "long back")


def _holidays(holidays) -> np.ndarray:
    return np.array([], dtype="datetime64[D]") if holidays is None else as_dates(holidays).ravel()


def _convention(convention: str) -> str:
    key = str(convention).lower().replace("_", " ").replace("-", " ")
    if key not in BUSINESS_DAY_CONVENTIONS:
        raise ValueError(f"unknown business-day convention {convention!r}; use one of {list(BUSINESS_DAY_CONVENTIONS)}")
    return key


def is_business_day(dates, holidays=None) -> np.ndarray:
    """Whether each date is a weekday and not in `holidays`.

    Parameters
    ----------
    dates : date-like or array-like of date-likes
    holidays : array-like of date-likes, optional

    Returns
    -------
    bool or ndarray of bool

    Examples
    --------
    >>> is_business_day(["2024-12-24", "2024-12-25", "2024-12-28"], holidays=["2024-12-25"]).tolist()
    [True, False, False]
    """
    return np.is_busday(as_dates(dates), holidays=_holidays(holidays))


def adjust_date(dates, convention: str = "following", holidays=None):
    """Move dates that are not business days by a business-day convention.

    Parameters
    ----------
    dates : date-like or array-like of date-likes
    convention : str
        ``"following"``, ``"modified following"``, ``"preceding"``,
        ``"modified preceding"`` or ``"unadjusted"`` (2006 ISDA
        Definitions, Sec. 4.12).
    holidays : array-like of date-likes, optional
        Holidays on top of Saturdays and Sundays.

    Returns
    -------
    datetime64[D] or ndarray of datetime64[D]

    Examples
    --------
    Saturday 30 September 2023: following crosses into October, so
    modified following goes back to Friday.

    >>> adjust_date("2023-09-30", "following"), adjust_date("2023-09-30", "modified following")
    (np.datetime64('2023-10-02'), np.datetime64('2023-09-29'))
    """
    roll = BUSINESS_DAY_CONVENTIONS[_convention(convention)]
    dates = as_dates(dates)
    if roll is None:
        return dates
    return np.busday_offset(dates, 0, roll=roll, holidays=_holidays(holidays))


def settlement_date(trade_date, lag: int = 2, holidays=None):
    """Settlement `lag` business days after the trade date (T + `lag`).

    A trade date that is not a business day is first rolled forward.

    Parameters
    ----------
    trade_date : date-like or array-like of date-likes
    lag : int
        Business days to settlement, at least 0.
    holidays : array-like of date-likes, optional

    Returns
    -------
    datetime64[D] or ndarray of datetime64[D]

    Examples
    --------
    A Friday trade settles T + 2 on Tuesday, or on Wednesday if Monday is a holiday.

    >>> settlement_date("2024-03-08"), settlement_date("2024-03-08", holidays=["2024-03-11"])
    (np.datetime64('2024-03-12'), np.datetime64('2024-03-13'))
    """
    if lag < 0:
        raise ValueError("lag must be nonnegative")
    return np.busday_offset(as_dates(trade_date), lag, roll="following", holidays=_holidays(holidays))


def coupon_schedule(
    start,
    end,
    freq: int = 2,
    stub: str = "short front",
    end_of_month: bool = False,
    convention: str = "unadjusted",
    holidays=None,
) -> Schedule:
    """Coupon dates from `start` to `end` at `freq` periods a year.

    Parameters
    ----------
    start, end : date-like
        Start of the first accrual period (a bond's dated date, a swap's
        effective date) and the maturity.
    freq : int
        Periods per year: 1, 2, 3, 4, 6 or 12.
    stub : str
        ``"short front"`` (the default) and ``"long front"`` roll back from
        `end`; ``"short back"`` and ``"long back"`` roll forward from
        `start`. A long stub absorbs the regular period next to it. On a
        whole number of periods every choice gives the same dates.
    end_of_month : bool
        Keep rolls from a month end on month ends (ISDA 2006, Sec. 4.13).
    convention : str
        Business-day convention for the adjusted dates; see :func:`adjust_date`.
    holidays : array-like of date-likes, optional
        Holidays on top of Saturdays and Sundays.

    Returns
    -------
    Schedule

    Examples
    --------
    A bond dated 1 February 2020 paying on 15 January and 15 July has a
    short first coupon:

    >>> s = coupon_schedule("2020-02-01", "2021-07-15")
    >>> [str(d) for d in s.unadjusted]
    ['2020-02-01', '2020-07-15', '2021-01-15', '2021-07-15']
    >>> [str(d) for d in coupon_schedule("2020-02-01", "2021-07-15", stub="long front").unadjusted]
    ['2020-02-01', '2021-01-15', '2021-07-15']
    """
    start, end = as_dates(start)[()], as_dates(end)[()]
    if end <= start:
        raise ValueError("end must be after start")
    if freq not in (1, 2, 3, 4, 6, 12):
        raise ValueError("freq must be 1, 2, 3, 4, 6 or 12")
    if stub not in STUBS:
        raise ValueError(f"unknown stub {stub!r}; use one of {list(STUBS)}")
    convention = _convention(convention)
    step = 12 // freq
    front = stub.endswith("front")
    anchor = end if front else start
    span = int((end.astype("datetime64[M]") - start.astype("datetime64[M]")).astype(np.int64)) // step + 2
    k = np.arange(-span, 1) if front else np.arange(0, span + 1)
    grid = add_months(anchor, step * np.concatenate([[k[0] - 1], k, [k[-1] + 1]]), end_of_month)
    lo, hi = np.searchsorted(grid, start, side="right") - 1, np.searchsorted(grid, end, side="left")
    quasi = grid[lo - 1 : hi + 2]
    interior = grid[(grid > start) & (grid < end)]
    if stub == "long front" and grid[lo] != start and len(interior) > 0:
        interior = interior[1:]
    if stub == "long back" and grid[hi] != end and len(interior) > 0:
        interior = interior[:-1]
    unadjusted = np.concatenate([[start], interior, [end]]).astype("datetime64[D]")
    hol = _holidays(holidays)
    return Schedule(
        unadjusted=unadjusted,
        dates=np.asarray(adjust_date(unadjusted, convention, hol)),
        quasi_dates=quasi,
        freq=freq,
        stub=stub,
        end_of_month=end_of_month,
        convention=convention,
        holidays=hol,
    )
