"""Calendar helpers for the day-count conventions, vectorised over numpy ``datetime64[D]``."""

from __future__ import annotations

import numpy as np

__all__ = ["as_dates", "date_parts", "is_leap_year", "days_in_year", "is_end_of_february"]


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
