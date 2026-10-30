"""Callable bonds on a short-rate tree, and the option-adjusted spread.

A callable bond lets its issuer buy it back at a fixed call price, which
the issuer does when rates have fallen and the bond is worth more than
that price. Its value is the straight bond minus a Bermudan call, found
by backward induction on Hull and White's (1994) trinomial tree
(:func:`~financialkit.interest_rate_models.systems.curve_fitted.hull_white_tree`):
on each call date the bond is worth the smaller of its continuation
value and the call price.

The option-adjusted spread (OAS) is the constant spread that, added to
every node rate, makes the tree's value match the market price. Spreads
measured on the straight cash flows (the Z-spread) mix the credit and
liquidity premium with the value of the call; the OAS takes the call out.

* :func:`callable_bond_price`: callable and straight values, and the
  rate below which the issuer calls on each call date.
* :func:`option_adjusted_spread`: the OAS, by :func:`scipy.optimize.brentq`.

References: J. Hull and A. White, "Numerical Procedures for Implementing
Term Structure Models I: Single-Factor Models," Journal of Derivatives
2(1) (1994) 7-16; J. C. Hull, *Options, Futures, and Other Derivatives*,
10th ed. (Pearson, 2018), Sec. 32.7; F. J. Fabozzi, *Bond Markets,
Analysis, and Strategies*, 9th ed. (Pearson, 2015), Ch. 18 (OAS).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import brentq

from financialkit._validation import positive
from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.fixed_income.core.base import coupon_times
from financialkit.interest_rate_models.core.base import CallableBondResult, TrinomialRateTree

__all__ = ["callable_bond_price", "option_adjusted_spread"]


def callable_bond_price(
    tree: TrinomialRateTree,
    maturity: float,
    coupon_rate: float,
    freq: int = 2,
    call_price: Optional[float] = None,
    first_call: float = 0.0,
    face: float = 1.0,
    spread: float = 0.0,
    last_call: Optional[float] = None,
) -> CallableBondResult:
    r"""Value a callable fixed-coupon bond by backward induction on a short-rate tree.

    The bond pays ``face * coupon_rate / freq`` on each coupon date and
    ``face`` at maturity. On every coupon date from `first_call` to
    `last_call` (by default up to but excluding maturity) the issuer may redeem it at `call_price` just after
    the coupon, so the ex-coupon value there is

    .. math:: V_i = \min\left(e^{-(r_i + s)\Delta t}\, E_i[V_{i+1}],\ K\right),

    with :math:`s` the `spread` and :math:`K` the call price. The straight
    bond is rolled back on the same tree without the minimum.

    Parameters
    ----------
    tree : TrinomialRateTree
        Every coupon date must be a node date.
    maturity : float
    coupon_rate : float
        Annual coupon rate.
    freq : int
        Coupons per year.
    call_price : float, optional
        Redemption price (default `face`, a call at par).
    first_call : float
        First call date; coupon dates before it are protected.
    face : float
    spread : float
        Spread added to every node rate, as in the OAS.
    last_call : float, optional
        Last call date; ``last_call = first_call`` makes the call European.

    Returns
    -------
    CallableBondResult

    Examples
    --------
    A 10-year 5% bond callable at par after 2 years, on a flat 5% curve.
    The call is worth about 3.1% of face:

    >>> from financialkit.interest_rate_models.systems.curve_fitted import hull_white_tree
    >>> tree = hull_white_tree(0.05, 0.1, 0.01, 10.0, 80)
    >>> res = callable_bond_price(tree, 10.0, 0.05, first_call=2.0, face=100.0)
    >>> round(res.straight_price, 2), round(res.option_value, 1)
    (99.51, 3.1)
    """
    positive(face=face)
    K = face if call_price is None else float(call_price)
    times = coupon_times(maturity, freq)
    steps = [tree.step_index(t) for t in times]
    n = steps[-1]
    coupon = face * coupon_rate / freq
    end = maturity - 1e-12 if last_call is None else last_call + 1e-12
    call_times = times[(times >= first_call - 1e-12) & (times < end)]
    call_steps = {tree.step_index(t) for t in call_times}
    coupon_steps = set(steps)
    rates = tree.alpha[:, None] + tree.levels * tree.dx
    reach = np.abs(tree.levels)
    boundary = {}
    callable_ = np.full(tree.centers.size, face + coupon)
    straight = callable_.copy()
    for i in range(n - 1, -1, -1):
        callable_ = tree.rollback(callable_, i, spread)
        straight = tree.rollback(straight, i, spread)
        if i in call_steps:
            called = (callable_ > K) & (reach <= min(i, tree.width))
            boundary[i] = float(rates[i][called].max()) if called.any() else np.nan
            callable_ = np.minimum(callable_, K)
        if i in coupon_steps:
            callable_ += coupon
            straight += coupon
    return CallableBondResult(
        price=float(callable_[tree.width]),
        straight_price=float(straight[tree.width]),
        call_times=call_times,
        call_boundary=np.array([boundary[tree.step_index(t)] for t in call_times]),
        spread=float(spread),
    )


def option_adjusted_spread(
    tree: TrinomialRateTree,
    price: float,
    maturity: float,
    coupon_rate: float,
    freq: int = 2,
    call_price: Optional[float] = None,
    first_call: float = 0.0,
    face: float = 1.0,
) -> float:
    r"""The option-adjusted spread: the spread over the tree's rates at which a callable bond is worth `price`.

    The tree is fitted to the benchmark (Treasury or swap) curve, so the
    call is valued at the benchmark's volatility, and what is left of the
    price gap is the spread. The bond's value falls with the spread, so
    :func:`scipy.optimize.brentq` finds it, widening its bracket as
    needed. Applied to a bond with no call it gives the Z-spread.

    Parameters
    ----------
    tree : TrinomialRateTree
    price : float
        Market (dirty) price, in the units of `face`.
    maturity, coupon_rate, freq, call_price, first_call, face
        As in :func:`callable_bond_price`.

    Returns
    -------
    float
        The OAS, continuously compounded.

    Examples
    --------
    The spread that reprices a bond valued at a 1% spread is 1%:

    >>> from financialkit.interest_rate_models.systems.curve_fitted import hull_white_tree
    >>> tree = hull_white_tree(0.05, 0.1, 0.01, 10.0, 80)
    >>> p = callable_bond_price(tree, 10.0, 0.05, first_call=2.0, spread=0.01).price
    >>> round(option_adjusted_spread(tree, p, 10.0, 0.05, first_call=2.0), 8)
    0.01
    """
    positive(price=price)

    def gap(s):
        return callable_bond_price(tree, maturity, coupon_rate, freq, call_price, first_call, face, s).price - price

    lo, hi = -0.05, 0.05
    while gap(lo) < 0:
        lo *= 2
    while gap(hi) > 0:
        hi *= 2
    return float(brentq(gap, lo, hi, xtol=DEFAULT_ATOL, maxiter=DEFAULT_MAX_ITER))
