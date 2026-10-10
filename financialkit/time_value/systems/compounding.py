"""Compound interest: accumulation, discounting and equivalent rates.

A rate compounded ``m`` times a year grows 1 to :math:`(1 + r/m)^{m t}`
by time ``t``; compounded continuously it grows to :math:`e^{r t}`, the
limit Jacob Bernoulli found in 1683 (*Acta Eruditorum*, 1690). Two rates
are *equivalent* when they produce the same growth, which is how a rate
quoted under one convention is restated under another (Hull, *Options,
Futures, and Other Derivatives*, 10th ed., 2018, Sec. 4.2). Also here:
Pacioli's rule of 72 (*Summa de arithmetica*, 1494, f. 181v) against the
exact doubling time, and compound-interest tables in the layout of
Richard Witt's *Arithmeticall Questions* (1613). Kellison, *The Theory
of Interest*, 3rd ed., 2009, Ch. 1, is the general reference.

``compounding=None`` means continuous compounding; a positive integer
``m`` means ``m`` periods per year.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from financialkit.time_value.core.base import CompoundingTable, _finalize

__all__ = [
    "future_value",
    "discount_factor",
    "continuous_rate",
    "equivalent_rate",
    "effective_annual_rate",
    "doubling_time",
    "rule_of_72",
    "compound_interest_table",
]


def _check_compounding(compounding: Optional[int]) -> None:
    if compounding is not None and (int(compounding) != compounding or compounding < 1):
        raise ValueError("compounding must be None (continuous) or a positive integer")


def continuous_rate(rate, compounding: Optional[int] = None):
    """Restate a rate as its continuously compounded equivalent, :math:`m \\ln(1 + r/m)`.

    Parameters
    ----------
    rate : float or array-like
        Annual rate, compounded ``compounding`` times a year.
    compounding : int or None
        ``None`` if `rate` is already continuous.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(continuous_rate(0.10, 1), 6)
    0.09531
    """
    _check_compounding(compounding)
    rate = np.asarray(rate, dtype=np.float64)
    if compounding is None:
        out = rate
    else:
        out = compounding * np.log1p(rate / compounding)
    return _finalize(out)


def discount_factor(rate, t, compounding: Optional[int] = None):
    """Value today of 1 paid at time ``t``.

    Parameters
    ----------
    rate : float or array-like
        Annual rate.
    t : float or array-like
        Time in years; broadcasts against `rate`.
    compounding : int or None
        Periods per year, or ``None`` for continuous compounding.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(discount_factor(0.05, 2.0), 6)
    0.904837
    >>> round(discount_factor(0.05, 2.0, compounding=1), 6)
    0.907029
    """
    return _finalize(np.exp(-np.multiply(continuous_rate(rate, compounding), np.asarray(t, dtype=np.float64))))


def future_value(present_value, rate, t, compounding: Optional[int] = None):
    """Accumulated value at time ``t`` of `present_value` invested today.

    Parameters
    ----------
    present_value : float or array-like
    rate : float or array-like
        Annual rate.
    t : float or array-like
        Time in years.
    compounding : int or None
        Periods per year, or ``None`` for continuous compounding.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Bernoulli's question: 1 at 100% a year, compounded ever more often.

    >>> [round(future_value(1.0, 1.0, 1.0, m), 6) for m in (1, 12, 365)]
    [2.0, 2.613035, 2.714567]
    >>> round(future_value(1.0, 1.0, 1.0), 6)
    2.718282
    """
    return _finalize(np.asarray(present_value, dtype=np.float64) / np.asarray(discount_factor(rate, t, compounding)))


def equivalent_rate(rate, from_compounding: Optional[int], to_compounding: Optional[int]):
    """Convert a rate between compounding frequencies at equal growth.

    Parameters
    ----------
    rate : float or array-like
    from_compounding, to_compounding : int or None
        Periods per year of the given and the requested rate (``None`` is
        continuous).

    Returns
    -------
    float or ndarray

    Examples
    --------
    Hull (10th ed.), Example 4.1: 10% compounded semiannually is 9.758%
    continuously compounded.

    >>> round(equivalent_rate(0.10, 2, None), 5)
    0.09758
    """
    _check_compounding(to_compounding)
    rc = continuous_rate(rate, from_compounding)
    if to_compounding is None:
        return rc
    return _finalize(to_compounding * np.expm1(np.asarray(rc) / to_compounding))


def effective_annual_rate(rate, compounding: Optional[int] = None):
    """Annual effective rate, :math:`(1 + r/m)^m - 1`, equivalent to `rate`.

    Parameters
    ----------
    rate : float or array-like
    compounding : int or None

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(effective_annual_rate(0.12, 12), 6)
    0.126825
    """
    return equivalent_rate(rate, compounding, 1)


def doubling_time(rate, compounding: Optional[int] = None):
    """Exact time, in years, for money to double at `rate`.

    Parameters
    ----------
    rate : float or array-like
        Positive annual rate.
    compounding : int or None

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(doubling_time(0.06, compounding=1), 4)
    11.8957
    """
    return _finalize(np.log(2.0) / np.asarray(continuous_rate(rate, compounding)))


def rule_of_72(rate, numerator: float = 72.0):
    """Pacioli's rule: money doubles in about ``72 / (100 * rate)`` years.

    The rule approximates the exact annual-compounding doubling time
    :math:`\\ln 2 / \\ln(1 + r)`; 72 rather than :math:`100 \\ln 2 \\approx
    69.3` corrects for the curvature of :math:`\\ln(1 + r)` near 8%, and
    72 has many divisors. Use ``numerator=69.3`` for continuous
    compounding.

    Parameters
    ----------
    rate : float or array-like
        Annual rate as a decimal.
    numerator : float
        The rule's constant.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> rule_of_72(0.06)
    12.0
    """
    return _finalize(numerator / (100.0 * np.asarray(rate, dtype=np.float64)))


def compound_interest_table(rate: float, n_periods: int) -> CompoundingTable:
    """Build a compound-interest table for periods ``1, ..., n_periods``.

    Witt (1613) printed, for each rate, the amount of 1, the present value
    of 1 and the values of an annuity of 1 per period; with them a clerk
    could price a loan, a lease or an annuity by look-up and multiplication.

    Parameters
    ----------
    rate : float
        Effective rate per period.
    n_periods : int

    Returns
    -------
    CompoundingTable

    Examples
    --------
    >>> table = compound_interest_table(0.10, 3)
    >>> table.amount.round(3).tolist(), table.annuity_present.round(4).tolist()
    ([1.1, 1.21, 1.331], [0.9091, 1.7355, 2.4869])
    """
    if n_periods < 1:
        raise ValueError("n_periods must be at least 1")
    k = np.arange(1, n_periods + 1)
    amount = (1.0 + rate) ** k
    discount = 1.0 / amount
    annuity_present = np.cumsum(discount)
    return CompoundingTable(
        rate=float(rate),
        periods=k,
        amount=amount,
        discount=discount,
        annuity_present=annuity_present,
        annuity_future=annuity_present * amount,
    )
