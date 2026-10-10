"""Illiquidity from daily data: Amihud's ratio and Kyle's lambda by regression.

Yakov Amihud (2002) measured illiquidity as the average absolute daily
return per dollar traded, the price response to order flow that Kyle's
lambda describes, computable from daily returns and volume alone. He
found that expected stock returns rise with it. Where signed order flow
is available, Kyle's lambda is the slope of price changes on it.

References: Y. Amihud, "Illiquidity and Stock Returns: Cross-Section and
Time-Series Effects," *J. Financial Markets* 5 (2002) 31-56; W. J.
Breen, L. S. Hodrick and R. A. Korajczyk, "Predicting Equity Liquidity,"
*Management Science* 48 (2002) 470-483.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.time_series.utils.numerics import OLSResult, ols

__all__ = ["amihud_illiquidity", "kyle_lambda"]


def amihud_illiquidity(returns: ArrayLike, dollar_volume: ArrayLike, scale: float = 1e6) -> float:
    """Amihud's ratio :math:`\\frac{1}{D}\\sum_d |r_d| / \\mathrm{DVol}_d`, over days with trading.

    Parameters
    ----------
    returns : array_like, shape (n,)
        Daily returns.
    dollar_volume : array_like, shape (n,)
        Daily dollar volume.
    scale : float
        Multiplier, conventionally :math:`10^6` (impact per million dollars).

    Returns
    -------
    float

    Examples
    --------
    >>> amihud_illiquidity([0.01, -0.02, 0.0], [2e6, 4e6, 0.0])
    0.005
    """
    r = np.asarray(returns, dtype=np.float64)
    dv = np.asarray(dollar_volume, dtype=np.float64)
    traded = dv > 0
    if not traded.any():
        raise ValueError("no day with positive volume")
    return float(scale * np.mean(np.abs(r[traded]) / dv[traded]))


def kyle_lambda(price_changes: ArrayLike, signed_volume: ArrayLike) -> OLSResult:
    """Regress price changes on signed order flow, :math:`\\Delta p = c + \\lambda y + e`.

    Returns
    -------
    OLSResult
        ``coef = [c, lambda]`` with standard errors.

    Examples
    --------
    >>> kyle_lambda([0.3, 0.0, 0.5], [2.0, -1.0, 4.0]).coef.round(10).tolist()
    [0.1, 0.1]
    """
    y = np.asarray(signed_volume, dtype=np.float64)
    return ols(np.asarray(price_changes, dtype=np.float64), np.column_stack([np.ones(y.size), y]))
