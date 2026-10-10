"""Market conventions and shared numerical tolerances used across financialkit.

Deliberately small: the calendar constants used to annualise daily data,
the day-count conventions every year-fraction function understands, and
the default convergence tolerances used by iterative solvers (implied
volatility, yield to maturity, curve bootstrapping, ...) so that "did this
converge" means the same thing everywhere in financialkit. There is no
currency or market data here; the year-fraction functions themselves live
in :mod:`financialkit.time_value`.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "CALENDAR_DAYS_PER_YEAR",
    "DayCount",
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "DEFAULT_MAX_ITER",
]

#: Trading days per year, used to annualise daily returns and volatilities.
TRADING_DAYS_PER_YEAR = 252

#: Calendar days per year (the ACT/365F denominator).
CALENDAR_DAYS_PER_YEAR = 365


class DayCount(str, Enum):
    """Day-count conventions for converting a date interval into a year fraction.

    Members compare equal to their market names, so ``DayCount("ACT/360")``
    and ``DayCount.ACT_360 == "ACT/360"`` both work.

    Examples
    --------
    >>> from financialkit.constants import DayCount
    >>> DayCount("30/360 US") is DayCount.THIRTY_360_US
    True
    """

    #: Actual days / 360 (money markets).
    ACT_360 = "ACT/360"
    #: Actual days / 365, fixed denominator.
    ACT_365F = "ACT/365F"
    #: 30/360 US (SIA bond basis), with the end-of-month rules.
    THIRTY_360_US = "30/360 US"
    #: 30E/360 (Eurobond basis).
    THIRTY_E_360 = "30E/360"
    #: Actual/actual ISDA: each calendar year's days over that year's length.
    ACT_ACT_ISDA = "ACT/ACT ISDA"


#: Default absolute tolerance for iterative/root-finding convergence checks.
DEFAULT_ATOL = 1e-10

#: Default relative tolerance for iterative/root-finding convergence checks.
DEFAULT_RTOL = 1e-8

#: Default maximum iteration count for iterative solvers/root finders.
DEFAULT_MAX_ITER = 1000
