"""Riskless discounting for the credit pricers: a flat rate or a ``fixed_income`` yield curve.

Every credit function that discounts takes ``r`` as either a continuously
compounded rate (a number, or an array that broadcasts) or a
:class:`~financialkit.fixed_income.YieldCurve`, so a CDS or a risky bond
can be priced off a bootstrapped term structure. A number gives exactly
the flat-rate discount factor :math:`e^{-rt}`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.fixed_income import CurveLike, YieldCurve

__all__ = ["discount_factors", "zero_rate_to"]


def discount_factors(r: ArrayLike | YieldCurve, t: ArrayLike) -> np.ndarray:
    """Riskless discount factors :math:`P(0, t)` from a flat rate or a yield curve.

    Parameters
    ----------
    r : array_like or YieldCurve
        Continuously compounded rate(s), broadcast against `t`, or a curve.
    t : array_like
        Times in years.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> from financialkit.fixed_income import FlatCurve
    >>> float(discount_factors(0.05, 2.0)) == float(discount_factors(FlatCurve(0.05), 2.0))
    True
    """
    t = np.asarray(t, dtype=np.float64)
    if isinstance(r, YieldCurve):
        return np.asarray(r.discount(t), dtype=np.float64)
    return np.exp(-np.asarray(r, dtype=np.float64) * t)


def zero_rate_to(r: CurveLike, T: float) -> float:
    """The continuously compounded zero rate to `T`: `r` itself, or the curve's.

    With deterministic rates a claim paid only at `T` sees the curve only
    through :math:`P(0, T)`, so the zero rate to `T` stands in for the
    whole curve exactly (Merton's equity and debt, for example).

    Parameters
    ----------
    r : float or YieldCurve
    T : float

    Returns
    -------
    float
    """
    if isinstance(r, YieldCurve):
        return float(r.zero_rate(T))
    return float(r)
