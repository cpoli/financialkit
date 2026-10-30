"""Calendar helpers for the day-count conventions, vectorised over numpy ``datetime64[D]``."""

from __future__ import annotations

import numpy as np

__all__ = ["as_dates", "date_parts", "is_leap_year", "days_in_year", "is_end_of_february", "is_end_of_month", "add_months"]


def as_dates(x) -> np.ndarray:
    """Convert dates to a ``datetime64[D]`` array.

    Parameters
    ----------
    x : date-like or array-like of date-likes
        :class:`datetime.date`, :class:`numpy.datetime64` or ISO
        ``"YYYY-MM-DD"`` strings.

    Returns
    -------
    ndarray of datetime64[D]

    Examples
    --------
    >>> as_dates(["2024-02-29", "2025-01-01"])
    array(['2024-02-29', '2025-01-01'], dtype='datetime64[D]')
    """
    return np.asarray(x, dtype="datetime64[D]")


def date_parts(d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dates into integer ``(year, month, day)`` arrays.

    Parameters
    ----------
    d : date-like or array-like of date-likes

    Returns
    -------
    year, month, day : ndarray of int

    Examples
    --------
    >>> [int(v) for v in date_parts("2024-02-29")]
    [2024, 2, 29]
    """
    d = as_dates(d)
    months = d.astype("datetime64[M]")
    year = months.astype("datetime64[Y]").astype(np.int64) + 1970
    month = months.astype(np.int64) % 12 + 1
    day = (d - months.astype("datetime64[D]")).astype(np.int64) + 1
    return year, month, day


def is_leap_year(year) -> np.ndarray:
    """Gregorian leap-year test, vectorised.

    Parameters
    ----------
    year : int or array-like of int

    Returns
    -------
    ndarray of bool
    """
    year = np.asarray(year)
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


def days_in_year(year) -> np.ndarray:
    """365 or 366 for each year.

    Parameters
    ----------
    year : int or array-like of int

    Returns
    -------
    ndarray of int
    """
    return np.where(is_leap_year(year), 366, 365)


def is_end_of_february(d) -> np.ndarray:
    """Whether each date is the last day of February.

    Parameters
    ----------
    d : date-like or array-like of date-likes

    Returns
    -------
    ndarray of bool
    """
    year, month, day = date_parts(d)
    return (month == 2) & (day == np.where(is_leap_year(year), 29, 28))


def is_end_of_month(d) -> np.ndarray:
    """Whether each date is the last day of its month.

    Parameters
    ----------
    d : date-like or array-like of date-likes

    Returns
    -------
    ndarray of bool

    Examples
    --------
    >>> is_end_of_month(["2024-02-28", "2024-02-29", "2023-02-28"]).tolist()
    [False, True, True]
    """
    d = as_dates(d)
    return (d + np.timedelta64(1, "D")).astype("datetime64[M]") != d.astype("datetime64[M]")


def add_months(d, n, end_of_month: bool = False) -> np.ndarray:
    """Shift dates by whole months, clamping the day to the target month's length.

    Parameters
    ----------
    d : date-like or array-like of date-likes
    n : int or array-like of int
        Months to add; negative moves back.
    end_of_month : bool
        If true, a date on the last day of its month lands on the last day
        of the target month (the end-of-month rule of the 2006 ISDA
        Definitions, Sec. 4.13).

    Returns
    -------
    ndarray of datetime64[D]

    Examples
    --------
    >>> add_months("2023-08-31", [-6, 6]).tolist()
    [datetime.date(2023, 2, 28), datetime.date(2024, 2, 29)]
    >>> add_months("2023-02-28", 3), add_months("2023-02-28", 3, end_of_month=True)
    (np.datetime64('2023-05-28'), np.datetime64('2023-05-31'))
    """
    d = as_dates(d)
    first = (d.astype("datetime64[M]") + np.asarray(n, dtype=np.int64).astype("timedelta64[M]")).astype("datetime64[D]")
    length = ((first.astype("datetime64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]") - first).astype(np.int64)
    day = date_parts(d)[2]
    day = np.where(end_of_month & is_end_of_month(d), length, np.minimum(day, length))
    return first + (day - 1).astype("timedelta64[D]")
