"""OIS discounting with a separate forecasting curve (Piterbarg, 2010).

Before 2008 one LIBOR curve both projected floating coupons and
discounted every cash flow. Once the LIBOR-OIS spread blew out, the
market recognised that a collateralized trade earns the collateral rate,
so its cash flows are discounted on the OIS curve, while LIBOR coupons
are projected from a separate forecasting curve. The OIS curve is
stripped from par OIS rates, whose floating leg is worth
:math:`1 - P(T)` exactly, so the stripping is
:func:`~financialkit.fixed_income.systems.curves.bootstrap_par_curve`.
The forecasting curve is then solved node by node so that every par
LIBOR swap, discounted on OIS, is worth zero. References:
V. Piterbarg, "Funding beyond Discounting: Collateral Agreements and
Derivatives Pricing," Risk 23(2) (2010), 97-102; M. Bianchetti, "Two
Curves, One Price," Risk 23(8) (2010), 66-72; J. C. Hull, *Options,
Futures, and Other Derivatives*, 9th ed. (Pearson, 2015), Sec. 9.3-9.4.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.fixed_income.core.base import CurveLike, MultiCurve, ZeroCurve, as_curve
from financialkit.fixed_income.systems.curves import bootstrap_par_curve

__all__ = ["bootstrap_forecast_curve", "bootstrap_multi_curve"]


def bootstrap_forecast_curve(discount: CurveLike, maturities, swap_rates, fixed_freq: int = 2, float_freq: int = 4) -> ZeroCurve:
    """Strip the forecasting curve that reprices par swaps discounted on `discount`.

    The curve is a :class:`~financialkit.fixed_income.core.base.ZeroCurve`
    of pseudo zero rates, linear between nodes and flat outside them, with
    one node per swap. Each node is found with
    :func:`scipy.optimize.brentq`, shortest swap first, so that the swap's
    floating leg (forward rates from this curve, discounted on
    `discount`) equals its fixed leg.

    Parameters
    ----------
    discount : YieldCurve or float
        The OIS discount curve.
    maturities : array_like, shape (n,)
        Distinct swap maturities in years, whole numbers of both periods.
    swap_rates : array_like, shape (n,)
        Par fixed rates of the swaps.
    fixed_freq, float_freq : int
        Payments per year on the fixed and floating legs.

    Returns
    -------
    ZeroCurve

    Examples
    --------
    With a single curve the forecasting curve is the discount curve:

    >>> from financialkit.fixed_income.core.base import FlatCurve
    >>> ois = FlatCurve(0.03)
    >>> K = MultiCurve(ois, ois).swap_rate([1.0, 5.0])
    >>> bool(np.allclose(bootstrap_forecast_curve(ois, [1.0, 5.0], K).rates, 0.03))
    True
    """
    discount = as_curve(discount)
    T = np.asarray(maturities, dtype=np.float64)
    K = np.broadcast_to(np.asarray(swap_rates, dtype=np.float64), T.shape)
    order = np.argsort(T)
    if np.any(np.diff(T[order]) <= 0):
        raise ValueError("maturities must be distinct")
    node_t: list[float] = []
    node_r: list[float] = []
    for k in order:

        def mispricing(r, k=k):
            curves = MultiCurve(discount, ZeroCurve(node_t + [float(T[k])], node_r + [r]))
            return curves.swap_rate(float(T[k]), fixed_freq, float_freq) - K[k]

        node_r.append(float(brentq(mispricing, -0.5, 2.0, xtol=DEFAULT_ATOL, maxiter=DEFAULT_MAX_ITER)))
        node_t.append(float(T[k]))
    return ZeroCurve(node_t, node_r)


def bootstrap_multi_curve(ois_maturities, ois_rates, swap_maturities, swap_rates, fixed_freq: int = 2, float_freq: int = 4, ois_freq: int = 1) -> MultiCurve:
    """OIS discount curve and LIBOR forecasting curve from par OIS and par swap rates.

    Parameters
    ----------
    ois_maturities, ois_rates : array_like
        Par OIS rates (fixed against compounded overnight, paid `ois_freq`
        times a year).
    swap_maturities, swap_rates : array_like
        Par rates of fixed-for-LIBOR swaps.
    fixed_freq, float_freq : int
        Payments per year on the LIBOR swaps' legs.
    ois_freq : int

    Returns
    -------
    MultiCurve

    Examples
    --------
    Annual swaps 20 basis points over OIS project 3-month LIBOR forwards
    about 20 basis points over the 3-month OIS forwards:

    >>> T = [1.0, 2.0, 5.0, 10.0]
    >>> curves = bootstrap_multi_curve(T, [0.02] * 4, T, [0.022] * 4, fixed_freq=1)
    >>> ois = MultiCurve(curves.discount, curves.discount)
    >>> round(1e4 * float(curves.forecast_rate(4.75, 5.0) - ois.forecast_rate(4.75, 5.0)))
    20
    """
    discount = bootstrap_par_curve(ois_maturities, ois_rates, freq=ois_freq)
    return MultiCurve(discount, bootstrap_forecast_curve(discount, swap_maturities, swap_rates, fixed_freq, float_freq))
