"""Interest-rate and currency swaps, valued as a long and a short bond.

The 1981 IBM-World Bank swap exchanged the payments on Swiss franc and
Deutsche mark debt for payments on US dollar debt: a fixed-for-fixed
currency swap, worth the difference between a foreign bond converted at
the spot rate and a domestic bond. The plain-vanilla interest-rate swap
that followed is the same idea in one currency, with a floating-rate leg
worth par on each reset date. References: J. C. Hull, *Options, Futures,
and Other Derivatives*, 9th ed. (Pearson, 2015), Ch. 7; C. W. Smith,
C. W. Smithson and L. M. Wakeman, "The Evolving Market for Swaps,"
Midland Corporate Finance Journal 3(4) (1986), 20-32.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from financialkit.fixed_income.core.base import CurveLike, SwapValuation, as_curve, coupon_times
from financialkit.fixed_income.systems.bonds import bond_cash_flows

__all__ = ["swap_annuity", "swap_rate", "interest_rate_swap_value", "currency_swap_value"]


def swap_annuity(curve: CurveLike, maturity: float, freq: int = 2) -> float:
    """Present value of 1 per year paid ``freq`` times a year to `maturity`: the fixed leg's PV01 times 10,000."""
    return float(np.sum(as_curve(curve).discount(coupon_times(maturity, freq))) / freq)


def swap_rate(curve: CurveLike, maturity, freq: int = 2):
    r"""Par swap rate: the fixed rate that makes a new swap worth zero.

    .. math:: K = \frac{1 - P(0, T)}{\frac{1}{m}\sum_i P(0, t_i)}

    It equals the par yield of a bond with the same schedule.

    Examples
    --------
    >>> from financialkit.fixed_income.core.base import FlatCurve
    >>> round(swap_rate(FlatCurve(np.log(1.03)), 5.0, freq=1), 10)
    0.03
    """
    return as_curve(curve).par_yield(maturity, freq)


def interest_rate_swap_value(
    curve: CurveLike,
    notional: float,
    fixed_rate: float,
    maturity: float,
    freq: int = 2,
    pay_fixed: bool = True,
    floating_rate: Optional[float] = None,
) -> SwapValuation:
    """Value of a fixed-for-floating interest-rate swap.

    The fixed leg is a bond with coupon `fixed_rate`. The floating leg is
    worth par immediately after a reset, so today it is worth the notional
    plus the already-fixed next coupon, discounted from the next payment
    date. Payments on both legs fall on ``maturity - k/freq``.

    Parameters
    ----------
    curve : YieldCurve or float
        Discount curve.
    notional : float
    fixed_rate : float
        Annual fixed rate.
    maturity : float
        Remaining life (years).
    freq : int
        Payments per year on both legs.
    pay_fixed : bool
        Value to the fixed-rate payer (``True``) or receiver.
    floating_rate : float, optional
        Annual rate already fixed for the next floating payment. May be
        omitted only on a reset date (maturity a whole number of periods),
        where the floating leg is worth exactly the notional.

    Returns
    -------
    SwapValuation

    Examples
    --------
    Hull (9th ed., Example 7.1): receive 8% and pay LIBOR on $100 million,
    1.25 years left, LIBOR zero rates 10%, 10.5% and 11% at 3, 9 and 15
    months, and 10.2% fixed for the next payment.

    >>> from financialkit.fixed_income.core.base import ZeroCurve
    >>> curve = ZeroCurve([0.25, 0.75, 1.25], [0.10, 0.105, 0.11])
    >>> v = interest_rate_swap_value(curve, 100.0, 0.08, 1.25, pay_fixed=False, floating_rate=0.102)
    >>> round(v.receive_leg, 3), round(v.pay_leg, 3), round(v.value, 3)
    (98.238, 102.505, -4.267)
    """
    curve = as_curve(curve)
    fixed_leg = bond_cash_flows(notional, fixed_rate, maturity, freq).present_value(curve)
    t1 = coupon_times(maturity, freq)[0]
    if floating_rate is None:
        if abs(t1 - 1.0 / freq) > 1e-9:
            raise ValueError("floating_rate is required between reset dates")
        floating_leg = float(notional)
    else:
        floating_leg = float(notional * (1.0 + floating_rate / freq) * curve.discount(t1))
    if pay_fixed:
        return SwapValuation(floating_leg - fixed_leg, floating_leg, fixed_leg, "pay_fixed")
    return SwapValuation(fixed_leg - floating_leg, fixed_leg, floating_leg, "receive_fixed")


def currency_swap_value(
    domestic_curve: CurveLike,
    foreign_curve: CurveLike,
    spot: float,
    domestic_notional: float,
    foreign_notional: float,
    domestic_rate: float,
    foreign_rate: float,
    maturity: float,
    freq: int = 1,
    receive_foreign: bool = True,
) -> SwapValuation:
    """Value, in domestic currency, of a fixed-for-fixed currency swap with exchange of principal.

    Each leg is a bond in its own currency discounted on its own curve; the
    foreign bond is converted at the spot rate.

    Parameters
    ----------
    domestic_curve, foreign_curve : YieldCurve or float
    spot : float
        Domestic currency per unit of foreign currency.
    domestic_notional, foreign_notional : float
    domestic_rate, foreign_rate : float
        Annual fixed coupon rates on each leg.
    maturity : float
    freq : int
    receive_foreign : bool
        Value to the party receiving the foreign leg (``True``) or paying it.

    Returns
    -------
    SwapValuation

    Examples
    --------
    Hull (9th ed., Example 7.3): receive 5% on 1,200 million yen and pay
    8% on $10 million annually for 3 years, with flat rates of 4% (yen) and
    9% (dollar) and 110 yen per dollar.

    >>> v = currency_swap_value(0.09, 0.04, 1 / 110, 10.0, 1200.0, 0.08, 0.05, 3.0)
    >>> round(v.value, 3)
    1.543
    """
    domestic = bond_cash_flows(domestic_notional, domestic_rate, maturity, freq).present_value(as_curve(domestic_curve))
    foreign = spot * bond_cash_flows(foreign_notional, foreign_rate, maturity, freq).present_value(as_curve(foreign_curve))
    if receive_foreign:
        return SwapValuation(foreign - domestic, foreign, domestic, "receive_foreign")
    return SwapValuation(domestic - foreign, domestic, foreign, "pay_foreign")
