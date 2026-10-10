"""Hasbrouck's vector autoregression: how much of a trade's price impact is permanent?

Joel Hasbrouck (1991) regressed quote revisions on current and lagged
signed trades and returns, and trades on lagged trades and returns. The
cumulative response of the quote to an unexpected trade settles at its
*permanent* impact, the information the trade carried; the overshoot
before that is transitory (inventory and bounce). The share of the
efficient price's variance driven by trade innovations measures how
much of the information reaches the price through trading.

Reference: J. Hasbrouck, "Measuring the Information Content of Stock
Trades," *J. Finance* 46 (1991) 179-207.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.market_microstructure.core.base import TradeImpactResult
from financialkit.time_series.utils.numerics import lag_matrix, ols

__all__ = ["hasbrouck_var"]


def _cumulative_response(coef_r: np.ndarray, coef_x: np.ndarray, lags: int, horizon: int, r0: float, x0: float) -> np.ndarray:
    a, b = coef_r[1 : 1 + lags], coef_r[1 + lags :]
    c, d = coef_x[:lags], coef_x[lags:]
    r = np.zeros(horizon + 1 + lags)
    x = np.zeros(horizon + 1 + lags)
    x[lags] = x0
    r[lags] = r0 + coef_r[0] * x0
    for t in range(lags + 1, horizon + 1 + lags):
        past_r = r[t - lags : t][::-1]
        past_x = x[t - lags : t][::-1]
        x[t] = c @ past_r + d @ past_x
        r[t] = coef_r[0] * x[t] + a @ past_r + b @ past_x
    return np.cumsum(r[lags:])


def hasbrouck_var(returns: ArrayLike, trades: ArrayLike, lags: int = 5, horizon: int = 20) -> TradeImpactResult:
    """Fit Hasbrouck's trade-impact VAR and its cumulative impulse response.

    .. math::

       r_t &= \\sum_{i=1}^p a_i r_{t-i} + \\sum_{i=0}^p b_i x_{t-i} + v_{1,t} \\\\
       x_t &= \\sum_{i=1}^p c_i r_{t-i} + \\sum_{i=1}^p d_i x_{t-i} + v_{2,t}

    Each equation is fitted by least squares
    (:func:`financialkit.time_series.utils.ols`) after demeaning.

    Parameters
    ----------
    returns : array_like, shape (n,)
        Midquote revisions (log or price changes) in trade time.
    trades : array_like, shape (n,)
        Trade signs or signed volume at the same events.
    lags : int
        Autoregressive order :math:`p`.
    horizon : int
        Number of events over which to accumulate the response.

    Returns
    -------
    TradeImpactResult
    """
    r = np.asarray(returns, dtype=np.float64)
    x = np.asarray(trades, dtype=np.float64)
    if r.shape != x.shape or r.ndim != 1 or r.size <= 3 * lags + 2:
        raise ValueError("returns and trades must be equal-length 1-D arrays longer than 3 * lags + 2")
    r = r - r.mean()
    x = x - x.mean()
    lr, lx = lag_matrix(r, lags), lag_matrix(x, lags)
    fit_r = ols(r[lags:], np.column_stack([x[lags:], lr, lx]))
    fit_x = ols(x[lags:], np.column_stack([lr, lx]))
    omega = np.cov(np.vstack([fit_r.residuals, fit_x.residuals]))
    resp_x = _cumulative_response(fit_r.coef, fit_x.coef, lags, horizon, 0.0, 1.0)
    resp_r = _cumulative_response(fit_r.coef, fit_x.coef, lags, horizon, 1.0, 0.0)
    a_r, a_x = resp_r[-1], resp_x[-1]
    var_w = a_r**2 * omega[0, 0] + a_x**2 * omega[1, 1] + 2 * a_r * a_x * omega[0, 1]
    return TradeImpactResult(
        coef_r=fit_r.coef,
        coef_x=fit_x.coef,
        impulse=resp_x,
        permanent_impact=float(a_x),
        trade_share=float(a_x**2 * omega[1, 1] / var_w),
        innovation_cov=omega,
    )
