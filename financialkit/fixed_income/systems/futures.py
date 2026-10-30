"""Treasury bond futures: conversion factors, the cheapest-to-deliver bond and the futures price.

The Chicago Board of Trade listed its Treasury bond future in August
1977. The short may deliver any bond from a basket, and receives the
futures price times the bond's *conversion factor* (its price per 1 of
face at a notional 6% yield, semiannual, with the maturity rounded down
to whole quarters) plus accrued interest. Conversion factors line the
bonds up only at 6%, so one bond is always cheapest to deliver, and
which one it is changes as yields move. References: J. C. Hull,
*Options, Futures, and Other Derivatives*, 9th ed. (Pearson, 2015),
Sec. 6.2; G. Burghardt, T. Belton, M. Lane and J. Papa, *The Treasury
Bond Basis*, 3rd ed. (McGraw-Hill, 2005), Ch. 1-2.
"""

from __future__ import annotations

import numpy as np

from financialkit.fixed_income.core.base import CashFlows, CurveLike, DeliveryBasis, as_curve

__all__ = ["conversion_factor", "cheapest_to_deliver", "treasury_futures_price"]


def conversion_factor(coupon_rate, maturity, notional_yield: float = 0.06, round_to_months: int = 3):
    """Conversion factor of a deliverable bond: its price per 1 of face at the notional yield.

    The maturity, measured from the first day of the delivery month, is
    rounded down to a multiple of `round_to_months` (three for the bond
    contract, one for the note contracts). With a whole number of
    half-years left the bond is priced on a coupon date; otherwise it is
    priced at the next coupon date, discounted back over the odd months at
    the notional yield, less the coupon accrued over the rest of the period.

    Parameters
    ----------
    coupon_rate : float or array_like
        Annual coupon rate, paid semiannually.
    maturity : float or array_like
        Years from the first day of the delivery month to maturity.
    notional_yield : float
        The contract's notional yield, compounded semiannually (6% since 2000).
    round_to_months : int
        Rounding of the maturity, in months.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Hull (9th ed., Sec. 6.2): a 10% bond with 20 years and 2 months left
    and an 8% bond with 18 years and 4 months left.

    >>> round(conversion_factor(0.10, 20 + 2 / 12), 4), round(conversion_factor(0.08, 18 + 4 / 12), 4)
    (1.4623, 1.2199)
    """
    if round_to_months not in (1, 3):
        raise ValueError("round_to_months must be 1 or 3")
    c = np.asarray(coupon_rate, dtype=np.float64) / 2.0
    T = np.asarray(maturity, dtype=np.float64)
    if np.any(T <= 0):
        raise ValueError("maturity must be positive")
    months = np.floor(T * 12.0 + 1e-9) // round_to_months * round_to_months
    n, m = months // 6, months % 6
    y = notional_yield / 2.0
    v = (1.0 + y) ** -n
    on_coupon_date = c * (1.0 - v) / y + v
    cf = np.where(m == 0, on_coupon_date, (c + on_coupon_date) * (1.0 + y) ** (-m / 6.0) - c * (6.0 - m) / 6.0)
    return float(cf) if cf.ndim == 0 else cf


def cheapest_to_deliver(futures_price: float, quoted_prices, conversion_factors) -> DeliveryBasis:
    """The deliverable bond that costs the short least to deliver.

    Delivering a bond costs its quoted price and earns the futures price
    times its conversion factor (accrued interest is paid either way), so
    the cheapest to deliver minimises ``quoted - futures * cf``.

    Parameters
    ----------
    futures_price : float
        Quoted futures price per 100 face.
    quoted_prices : array_like, shape (n,)
        Quoted (clean) prices of the deliverable bonds per 100 face.
    conversion_factors : array_like, shape (n,)

    Returns
    -------
    DeliveryBasis

    Examples
    --------
    Hull (9th ed., Sec. 6.2): futures at 93-08 and three bonds.

    >>> basis = cheapest_to_deliver(93.25, [99.50, 143.50, 119.75], [1.0382, 1.5188, 1.2615])
    >>> basis.costs.round(2).tolist(), basis.cheapest
    ([2.69, 1.87, 2.12], 1)
    """
    prices = np.asarray(quoted_prices, dtype=np.float64)
    cfs = np.asarray(conversion_factors, dtype=np.float64)
    if prices.ndim != 1 or prices.shape != cfs.shape or prices.size == 0:
        raise ValueError("quoted_prices and conversion_factors must be 1-D arrays of the same nonzero length")
    return DeliveryBasis(float(futures_price), prices, cfs)


def treasury_futures_price(
    curve: CurveLike,
    quoted_price: float,
    accrued: float,
    coupons: CashFlows,
    delivery: float,
    accrued_at_delivery: float,
    conversion_factor: float = 1.0,
) -> float:
    """Quoted futures price, assuming the cheapest-to-deliver bond and the delivery date are known.

    The futures cash price is the forward price of the bond: its cash
    (dirty) price less the present value of the coupons paid before
    delivery, grown to the delivery date on `curve`. Subtracting the
    accrued interest at delivery and dividing by the conversion factor
    gives the quoted futures price.

    Parameters
    ----------
    curve : YieldCurve or float
        Financing (repo) curve.
    quoted_price : float
        Quoted (clean) price of the cheapest-to-deliver bond today.
    accrued : float
        Its accrued interest today.
    coupons : CashFlows
        Its remaining coupons; those after `delivery` are ignored.
    delivery : float
        Years to delivery.
    accrued_at_delivery : float
        Its accrued interest on the delivery date.
    conversion_factor : float

    Returns
    -------
    float

    Examples
    --------
    Hull (9th ed., Sec. 6.2): a 12% bond quoted at 115 with conversion
    factor 1.6, 60 days into a 182-day coupon period, a coupon in 122
    days, delivery in 270 days, 148 days into the next 183-day period,
    and a flat 10% rate.

    >>> coupons = CashFlows([122 / 365, 305 / 365], [6.0, 6.0])
    >>> round(treasury_futures_price(0.10, 115.0, 6 * 60 / 182, coupons, 270 / 365, 6 * 148 / 183, 1.6), 2)
    71.79
    """
    if delivery <= 0:
        raise ValueError("delivery must be positive")
    curve = as_curve(curve)
    before = coupons.times < delivery
    income = CashFlows(coupons.times[before], coupons.amounts[before]).present_value(curve)
    cash_futures = (quoted_price + accrued - income) / float(curve.discount(delivery))
    return float((cash_futures - accrued_at_delivery) / conversion_factor)
