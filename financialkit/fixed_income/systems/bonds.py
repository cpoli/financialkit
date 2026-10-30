"""Bond cash flows, prices, yield to maturity, accrued interest and STRIPS.

A bond is a :class:`~financialkit.fixed_income.core.base.CashFlows`
stream, so pricing it off a single yield (Homer and Leibowitz's yield to
maturity) and off a zero curve (the STRIPS view of a bond as a portfolio
of zero-coupon bonds) are the same discounting under two different
discount functions. Yield to maturity inverts the price with
:func:`scipy.optimize.brentq`. References: S. Homer and M. L. Leibowitz,
*Inside the Yield Book* (Prentice-Hall, 1972); J. C. Hull, *Options,
Futures, and Other Derivatives*, 9th ed. (Pearson, 2015), Ch. 4;
F. J. Fabozzi, *Bond Markets, Analysis, and Strategies*, 9th ed.
(Pearson, 2015), Ch. 2-3.

Prices are per the ``face`` passed (100 by default) and are *dirty*
(full) prices unless a function says ``clean``. Times are year fractions
from settlement; a maturity that is not a whole number of coupon periods
puts settlement inside the first coupon period.

Every bond function also takes a dated
:class:`~financialkit.time_value.Schedule` and a settlement date in place
of `maturity` and `freq`. Odd first and last coupons are then paid,
accrued and discounted the SIA way (Mayle, *Standard Securities
Calculation Methods*, 3rd ed., 1993): a stub's coupon is its share of the
quasi-coupon periods it spans, accrual runs on the unadjusted dates, and
the time to each payment is the number of quasi-coupon periods to it over
`freq`. Discounting with ``compounding=freq`` is then the street yield.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import brentq

from financialkit._deprecation import deprecated_alias
from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.fixed_income.core.base import CashFlows, YieldCurve, ZeroCurve, coupon_times
from financialkit.time_value import Schedule, discount_factor, year_fraction
from financialkit.time_value.utils.dates import as_dates

__all__ = [
    "bond_cash_flows",
    "price_from_yield",
    "bond_price",
    "yield_to_maturity",
    "bond_accrued_interest",
    "accrued_interest",
    "clean_price",
    "zero_coupon_bond_price",
    "strips_decomposition",
    "strips_value",
    "strips_curve",
]


def _scheduled_bond(face: float, coupon_rate: float, schedule: Schedule, settlement, day_count) -> tuple[CashFlows, float]:
    """Remaining cash flows (times from `settlement`) and the accrued interest of a bond on a dated schedule."""
    if settlement is None:
        raise ValueError("settlement is required with a schedule")
    s = as_dates(settlement)[()]
    d = schedule.unadjusted
    if not d[0] <= s < d[-1]:
        raise ValueError("settlement must be on or after the schedule's start and before its end")
    i = int(np.searchsorted(d, s, side="right"))
    starts, ends = d[i - 1 : -1], d[i:]

    def fraction(a, b):
        if day_count is None:
            return np.asarray(schedule.periods_between(a, b), dtype=np.float64) / schedule.freq
        return np.asarray(year_fraction(a, b, day_count), dtype=np.float64)

    times = np.asarray(schedule.periods_between(s, ends, day_count), dtype=np.float64).reshape(-1) / schedule.freq
    accrued = float(face * coupon_rate * fraction(d[i - 1], s))
    if coupon_rate == 0.0:
        return CashFlows(times[-1:], np.array([float(face)])), accrued
    amounts = face * coupon_rate * fraction(starts, ends).reshape(-1)
    amounts[-1] += face
    return CashFlows(times, amounts), accrued


def _maturity(maturity: Optional[float], schedule: Optional[Schedule]) -> float:
    if (maturity is None) == (schedule is None):
        raise ValueError("pass either maturity or schedule")
    return float(maturity) if maturity is not None else 0.0


def bond_cash_flows(
    face: float,
    coupon_rate: float,
    maturity: Optional[float] = None,
    freq: int = 2,
    *,
    schedule: Optional[Schedule] = None,
    settlement=None,
    day_count=None,
) -> CashFlows:
    """Cash flows of a fixed-coupon bullet bond.

    Parameters
    ----------
    face : float
        Principal repaid at maturity.
    coupon_rate : float
        Annual coupon rate (decimal); each coupon is ``face * coupon_rate / freq``.
        Zero gives a zero-coupon bond with a single payment.
    maturity : float, optional
        Years to maturity; omit when passing `schedule`.
    freq : int
        Coupons per year.
    schedule : Schedule, optional
        Dated coupon schedule, from :func:`financialkit.time_value.coupon_schedule`,
        in place of `maturity` and `freq`.
    settlement : date-like, optional
        Settlement date, on or after the schedule's start and before its
        end; required with `schedule`.
    day_count : DayCount, str or None
        With `schedule`, how coupon periods are measured: ``None`` (the
        default) is actual/actual by coupon period (ICMA, SIA's Treasury
        basis); a :class:`~financialkit.constants.DayCount` such as
        ``"30/360 US"`` counts days that way, and each coupon is
        ``face * coupon_rate * year_fraction``.

    Returns
    -------
    CashFlows
        With `schedule`, only the payments after `settlement`, at times in
        years from it.

    Examples
    --------
    >>> cf = bond_cash_flows(100.0, 0.06, 1.5)
    >>> cf.times.tolist(), cf.amounts.tolist()
    ([0.5, 1.0, 1.5], [3.0, 3.0, 103.0])

    A bond dated 15 October 2008 with a short first coupon on 1 March 2009
    pays 137/181 of a regular coupon then, the share of the quasi-coupon
    period from 1 September 2008 that it accrues over:

    >>> from financialkit.time_value import coupon_schedule
    >>> s = coupon_schedule("2008-10-15", "2010-03-01")
    >>> cf = bond_cash_flows(100.0, 0.0785, schedule=s, settlement="2008-11-11")
    >>> cf.amounts.round(6).tolist(), round(float(cf.times[0]), 6)
    ([2.970856, 3.925, 103.925], 0.303867)
    """
    maturity = _maturity(maturity, schedule)
    if schedule is not None:
        return _scheduled_bond(face, coupon_rate, schedule, settlement, day_count)[0]
    if coupon_rate == 0.0:
        return CashFlows(np.array([float(maturity)]), np.array([float(face)]))
    times = coupon_times(maturity, freq)
    amounts = np.full(times.shape, face * coupon_rate / freq)
    amounts[-1] += face
    return CashFlows(times, amounts)


def price_from_yield(cash_flows: CashFlows, y, compounding: Optional[int] = None):
    """Present value of `cash_flows` discounted at a single yield `y`; broadcasts over `y`.

    Parameters
    ----------
    cash_flows : CashFlows
    y : float or array_like
        Yield (decimal).
    compounding : int or None
        Compounding frequency of `y`; ``None`` for continuous.

    Returns
    -------
    float or ndarray
    """
    y = np.asarray(y, dtype=np.float64)
    pv = np.sum(cash_flows.amounts * discount_factor(y[..., None], cash_flows.times, compounding), axis=-1)
    return float(pv) if pv.ndim == 0 else pv


def bond_price(
    face: float,
    coupon_rate: float,
    maturity: Optional[float],
    y,
    freq: int = 2,
    compounding: Optional[int] = None,
    *,
    schedule: Optional[Schedule] = None,
    settlement=None,
    day_count=None,
):
    """Dirty price of a fixed-coupon bond at yield `y`; broadcasts over `y`.

    Hull's convention (continuous compounding, ``compounding=None``) is the
    default; pass ``compounding=freq`` for the street convention of a yield
    compounded at the coupon frequency. For a dated bond pass
    ``maturity=None`` and a `schedule`, `settlement` and `day_count` as in
    :func:`bond_cash_flows`.

    Examples
    --------
    Hull (9th ed., Sec. 4.4): a 2-year 6% semiannual bond yielding 6.76%
    continuously compounded is worth 98.39.

    >>> round(bond_price(100.0, 0.06, 2.0, 0.0676), 2)
    98.39
    >>> round(bond_price(100.0, 0.05, 10.0, 0.05, compounding=2), 10)  # at par
    100.0
    """
    cf = bond_cash_flows(face, coupon_rate, maturity, freq, schedule=schedule, settlement=settlement, day_count=day_count)
    return price_from_yield(cf, y, compounding)


def yield_to_maturity(price: float, cash_flows: CashFlows, compounding: Optional[int] = None, tol: float = DEFAULT_ATOL) -> float:
    """The single discount rate that equates `cash_flows`' present value to `price`.

    The price is strictly decreasing in the yield for positive cash flows,
    so the root is unique; it is bracketed by expanding an interval and
    found with :func:`scipy.optimize.brentq`.

    Parameters
    ----------
    price : float
        Dirty price.
    cash_flows : CashFlows
        Positive cash flows.
    compounding : int or None
        Compounding frequency of the returned yield; ``None`` for continuous.
    tol : float
        Absolute tolerance on the yield.

    Returns
    -------
    float

    Examples
    --------
    >>> cf = bond_cash_flows(100.0, 0.06, 2.0)
    >>> round(yield_to_maturity(98.39, cf), 4)  # Hull, 9th ed., Sec. 4.4
    0.0676
    """
    if price <= 0:
        raise ValueError("price must be positive")
    if np.any(cash_flows.amounts < 0):
        raise ValueError("yield_to_maturity needs nonnegative cash flows")
    f = lambda y: price_from_yield(cash_flows, y, compounding) - price
    if compounding is None:
        lo = -1.0
        while f(lo) < 0:
            lo -= 1.0
            if lo < -50:
                raise ValueError("price is above any attainable present value")
    else:
        lo = -0.999 * float(compounding)
    hi = 1.0
    while f(hi) > 0:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError("price is below any attainable present value")
    return float(brentq(f, lo, hi, xtol=tol, maxiter=DEFAULT_MAX_ITER))


def bond_accrued_interest(
    face: float,
    coupon_rate: float,
    maturity: Optional[float] = None,
    freq: int = 2,
    *,
    schedule: Optional[Schedule] = None,
    settlement=None,
    day_count=None,
) -> float:
    """Coupon interest earned since the last coupon date, owed by the buyer to the seller.

    This is the year-fraction version, matching :func:`bond_price`: the
    bond is described by its time to `maturity` in years and a regular
    coupon frequency, so settlement lies ``maturity mod 1/freq`` before the
    next coupon. For accrual between calendar dates under a day-count
    convention, use :func:`financialkit.time_value.accrued_interest`.
    This function was ``fixed_income.accrued_interest`` in 0.1.0; that
    name remains as a deprecated alias.

    With a dated `schedule`, interest accrues from the last unadjusted
    coupon date (or the start, in a first period) to `settlement`, and a
    stub accrues its share of the quasi-coupon periods it spans.

    Parameters
    ----------
    face : float
    coupon_rate : float
        Annual coupon rate.
    maturity : float, optional
        Time from settlement to maturity in years; omit when passing `schedule`.
    freq : int
        Coupons per year.
    schedule : Schedule, optional
    settlement : date-like, optional
    day_count : DayCount, str or None
        As in :func:`bond_cash_flows`.

    Returns
    -------
    float

    Examples
    --------
    >>> bond_accrued_interest(100.0, 0.08, 1.375, freq=2)  # a quarter of the way into a 4.0 coupon
    1.0

    Hull (10th ed.), Sec. 6.1: a 10% Treasury paying on 1 March and 1
    September has accrued 124/184 of a 5.00 coupon by 3 July 2018.

    >>> from financialkit.time_value import coupon_schedule
    >>> s = coupon_schedule("2017-09-01", "2020-09-01")
    >>> round(bond_accrued_interest(100.0, 0.10, schedule=s, settlement="2018-07-03"), 4)
    3.3696
    """
    maturity = _maturity(maturity, schedule)
    if schedule is not None:
        return _scheduled_bond(face, coupon_rate, schedule, settlement, day_count)[1]
    times = coupon_times(maturity, freq)
    elapsed = 1.0 / freq - times[0]
    return float(face * coupon_rate / freq * elapsed * freq)


accrued_interest = deprecated_alias(bond_accrued_interest, "accrued_interest", "financialkit.fixed_income.bond_accrued_interest")


def clean_price(
    face: float,
    coupon_rate: float,
    maturity: Optional[float],
    y,
    freq: int = 2,
    compounding: Optional[int] = None,
    *,
    schedule: Optional[Schedule] = None,
    settlement=None,
    day_count=None,
):
    """Quoted (clean) price: the dirty price less accrued interest; broadcasts over `y`.

    The dirty price jumps down by a coupon on each payment date and climbs
    between them; the clean price removes that sawtooth. A dated bond takes
    ``maturity=None`` and `schedule`, `settlement` and `day_count` as in
    :func:`bond_cash_flows`.

    Examples
    --------
    The SIA price of a bond with a short first coupon: dated 15 October
    2008, first coupon 1 March 2009, maturing 1 March 2021, 7.85%
    semiannual, settling 11 November 2008 at a 6.25% yield. Excel's
    ``ODDFPRICE``, which implements the SIA formula, gives 113.597717.

    >>> from financialkit.time_value import coupon_schedule
    >>> s = coupon_schedule("2008-10-15", "2021-03-01")
    >>> round(clean_price(100.0, 0.0785, None, 0.0625, compounding=2, schedule=s, settlement="2008-11-11"), 6)
    113.597717
    """
    dirty = bond_price(face, coupon_rate, maturity, y, freq, compounding, schedule=schedule, settlement=settlement, day_count=day_count)
    return dirty - bond_accrued_interest(face, coupon_rate, maturity, freq, schedule=schedule, settlement=settlement, day_count=day_count)


def zero_coupon_bond_price(T, y, face: float = 100.0, compounding: Optional[int] = None):
    """Price of a zero-coupon bond paying `face` at `T`, at yield `y`; broadcasts.

    Examples
    --------
    >>> round(float(zero_coupon_bond_price(1.0, 0.10536)), 2)  # Hull, Table 4.3
    90.0
    """
    return face * discount_factor(y, T, compounding)


def strips_decomposition(cash_flows: CashFlows) -> list:
    """Split a coupon bond into the zero-coupon bonds (STRIPS) it is a portfolio of.

    Returns
    -------
    list of CashFlows
        One single-payment stream per payment date, in date order.

    Examples
    --------
    >>> [s.amounts.tolist() for s in strips_decomposition(bond_cash_flows(100.0, 0.04, 1.0))]
    [[2.0], [102.0]]
    """
    return [CashFlows(np.array([t]), np.array([a])) for t, a in zip(cash_flows.times, cash_flows.amounts, strict=True)]


def strips_value(cash_flows: CashFlows, strip_prices, face: float = 100.0) -> float:
    """Cost of replicating `cash_flows` with STRIPS quoted at `strip_prices` per `face` of each payment date.

    Comparing this with the bond's market price is the stripping /
    reconstitution trade: if the strips are worth more, buy the bond and
    strip it; if less, buy the strips and reconstitute.

    Parameters
    ----------
    cash_flows : CashFlows
    strip_prices : array_like or YieldCurve
        Prices per `face` of zero-coupon bonds maturing on each payment
        date, or a curve to price them on.
    face : float

    Returns
    -------
    float
    """
    if isinstance(strip_prices, YieldCurve):
        return cash_flows.present_value(strip_prices)
    prices = np.asarray(strip_prices, dtype=np.float64)
    if prices.shape != cash_flows.times.shape:
        raise ValueError("need one strip price per payment date")
    return float(np.sum(cash_flows.amounts * prices / face))


def strips_curve(times, strip_prices, face: float = 100.0) -> ZeroCurve:
    """Zero curve implied by STRIPS prices: :math:`r(t) = -\\ln(p/F)/t`.

    Examples
    --------
    >>> curve = strips_curve([0.5, 1.0], [97.5, 95.0])
    >>> round(float(curve.zero_rate(1.0)), 6)
    0.051293
    """
    times = np.asarray(times, dtype=np.float64)
    prices = np.asarray(strip_prices, dtype=np.float64)
    return ZeroCurve(times, -np.log(prices / face) / times)
