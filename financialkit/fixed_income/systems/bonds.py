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
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import brentq

from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.fixed_income.core.base import CashFlows, YieldCurve, ZeroCurve, coupon_times
from financialkit.time_value import discount_factor

__all__ = [
    "bond_cash_flows",
    "price_from_yield",
    "bond_price",
    "yield_to_maturity",
    "accrued_interest",
    "clean_price",
    "zero_coupon_bond_price",
    "strips_decomposition",
    "strips_value",
    "strips_curve",
]


def bond_cash_flows(face: float, coupon_rate: float, maturity: float, freq: int = 2) -> CashFlows:
    """Cash flows of a fixed-coupon bullet bond.

    Parameters
    ----------
    face : float
        Principal repaid at maturity.
    coupon_rate : float
        Annual coupon rate (decimal); each coupon is ``face * coupon_rate / freq``.
        Zero gives a zero-coupon bond with a single payment.
    maturity : float
        Years to maturity.
    freq : int
        Coupons per year.

    Returns
    -------
    CashFlows

    Examples
    --------
    >>> cf = bond_cash_flows(100.0, 0.06, 1.5)
    >>> cf.times.tolist(), cf.amounts.tolist()
    ([0.5, 1.0, 1.5], [3.0, 3.0, 103.0])
    """
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


def bond_price(face: float, coupon_rate: float, maturity: float, y, freq: int = 2, compounding: Optional[int] = None):
    """Dirty price of a fixed-coupon bond at yield `y`; broadcasts over `y`.

    Hull's convention (continuous compounding, ``compounding=None``) is the
    default; pass ``compounding=freq`` for the street convention of a yield
    compounded at the coupon frequency.

    Examples
    --------
    Hull (9th ed., Sec. 4.4): a 2-year 6% semiannual bond yielding 6.76%
    continuously compounded is worth 98.39.

    >>> round(bond_price(100.0, 0.06, 2.0, 0.0676), 2)
    98.39
    >>> round(bond_price(100.0, 0.05, 10.0, 0.05, compounding=2), 10)  # at par
    100.0
    """
    return price_from_yield(bond_cash_flows(face, coupon_rate, maturity, freq), y, compounding)


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


def accrued_interest(face: float, coupon_rate: float, maturity: float, freq: int = 2) -> float:
    """Coupon interest earned since the last coupon date, owed by the buyer to the seller.

    The accrual fraction is the elapsed share of the current coupon period
    measured in years; how calendar dates become those years is a day-count
    question for :mod:`financialkit.time_value`; for accrual between calendar dates
    use :func:`financialkit.time_value.accrued_interest`.

    Examples
    --------
    >>> accrued_interest(100.0, 0.08, 1.375, freq=2)  # a quarter of the way into a 4.0 coupon
    1.0
    """
    times = coupon_times(maturity, freq)
    elapsed = 1.0 / freq - times[0]
    return float(face * coupon_rate / freq * elapsed * freq)


def clean_price(face: float, coupon_rate: float, maturity: float, y, freq: int = 2, compounding: Optional[int] = None):
    """Quoted (clean) price: the dirty price less accrued interest; broadcasts over `y`.

    The dirty price jumps down by a coupon on each payment date and climbs
    between them; the clean price removes that sawtooth.
    """
    dirty = bond_price(face, coupon_rate, maturity, y, freq, compounding)
    return dirty - accrued_interest(face, coupon_rate, maturity, freq)


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
