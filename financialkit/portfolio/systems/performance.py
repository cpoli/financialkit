"""Risk-adjusted performance: Sharpe (1966), Grinold (1989) and Sortino (1994).

Sharpe's *reward-to-variability ratio* divides a fund's mean excess
return by its volatility; Lo (2002) gave its sampling error. Grinold's
fundamental law of active management says a manager's information ratio
is roughly the skill of each forecast, the information coefficient, times
the square root of the number of independent forecasts:
:math:`IR \\approx IC\\sqrt{BR}`. Sortino and Price replaced volatility by
the downside deviation below a minimum acceptable return, so upside
surprises are not penalized.

References: W. F. Sharpe, "Mutual Fund Performance," *J. Business* 39
(1966) 119-138; A. W. Lo, "The Statistics of Sharpe Ratios," *Financial
Analysts J.* 58 (2002) 36-52; R. C. Grinold, "The Fundamental Law of
Active Management," *J. Portfolio Mgmt.* 15 (1989) 30-37; F. A. Sortino
and L. N. Price, "Performance Measurement in a Downside Risk Framework,"
*J. Investing* 3 (1994) 59-64.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.portfolio.core.base import ActiveManagementResult

__all__ = [
    "sharpe_ratio",
    "sharpe_ratio_se",
    "information_ratio",
    "downside_deviation",
    "sortino_ratio",
    "fundamental_law",
    "simulate_active_management",
]


def _scalar(x: np.ndarray) -> np.ndarray | float:
    return float(x) if np.ndim(x) == 0 else x


def sharpe_ratio(returns: ArrayLike, rf: float = 0.0, periods_per_year: float = 1.0) -> np.ndarray | float:
    """Sharpe ratio: mean excess return over its standard deviation, annualized.

    Parameters
    ----------
    returns : array_like, shape (n,) or (n, k)
        Periodic returns; each column is a separate series.
    rf : float
        Riskless return per period.
    periods_per_year : float
        Scales the per-period ratio by its square root (``252`` for daily, ``12`` for monthly).

    Returns
    -------
    float or ndarray, shape (k,)

    Examples
    --------
    >>> round(sharpe_ratio([0.02, -0.01, 0.03, 0.0], periods_per_year=12), 4)
    1.8974
    """
    X = np.asarray(returns, dtype=np.float64) - rf
    return _scalar(X.mean(axis=0) / X.std(axis=0, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio_se(returns: ArrayLike, rf: float = 0.0, periods_per_year: float = 1.0) -> np.ndarray | float:
    """Lo's (2002) standard error of the Sharpe ratio for independent returns.

    The per-period estimate has asymptotic standard error
    :math:`\\sqrt{(1 + \\tfrac12 SR^2)/n}`; annualizing multiplies it by
    :math:`\\sqrt{q}` like the ratio itself.

    Parameters
    ----------
    returns, rf, periods_per_year
        As in :func:`sharpe_ratio`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Ten years of monthly returns pin down an annual Sharpe ratio of 0.5
    only to about plus or minus 0.32.

    >>> import numpy as np
    >>> r = np.random.default_rng(0).normal(0.005, 0.04, 120)
    >>> round(sharpe_ratio_se(r, periods_per_year=12), 2)
    0.32
    """
    X = np.asarray(returns, dtype=np.float64)
    sr = np.asarray(sharpe_ratio(X, rf))  # per period
    return _scalar(np.sqrt((1.0 + 0.5 * sr**2) / X.shape[0]) * np.sqrt(periods_per_year))


def information_ratio(returns: ArrayLike, benchmark: ArrayLike, periods_per_year: float = 1.0) -> np.ndarray | float:
    """Mean active return over tracking error, annualized.

    Parameters
    ----------
    returns : array_like, shape (n,) or (n, k)
    benchmark : array_like, shape (n,)
    periods_per_year : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(information_ratio([0.03, 0.01, 0.02], [0.02, 0.01, 0.0]), 4)
    1.0
    """
    R = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    active = R - (b[:, None] if R.ndim == 2 else b)
    return sharpe_ratio(active, 0.0, periods_per_year)


def downside_deviation(returns: ArrayLike, target: float = 0.0, periods_per_year: float = 1.0) -> np.ndarray | float:
    """Root mean square shortfall below a minimum acceptable return.

    :math:`DD = \\sqrt{\\tfrac1n \\sum_t \\min(R_t - \\tau, 0)^2}`, the
    square root of the second lower partial moment, annualized by
    :math:`\\sqrt{q}`.

    Parameters
    ----------
    returns : array_like, shape (n,) or (n, k)
    target : float
        Minimum acceptable return per period, :math:`\\tau`.
    periods_per_year : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> downside_deviation([0.03, -0.04, 0.01, -0.02])
    0.022360679774997897
    """
    X = np.asarray(returns, dtype=np.float64)
    return _scalar(np.sqrt(np.mean(np.minimum(X - target, 0.0) ** 2, axis=0) * periods_per_year))


def sortino_ratio(returns: ArrayLike, target: float = 0.0, periods_per_year: float = 1.0) -> np.ndarray | float:
    """Sortino ratio: mean return above the target over the downside deviation, annualized.

    Parameters
    ----------
    returns : array_like, shape (n,) or (n, k)
    target : float
        Minimum acceptable return per period.
    periods_per_year : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(sortino_ratio([0.03, -0.04, 0.01, -0.02]), 4)
    -0.2236
    """
    X = np.asarray(returns, dtype=np.float64)
    excess = (X.mean(axis=0) - target) * periods_per_year
    return _scalar(excess / np.asarray(downside_deviation(X, target, periods_per_year)))


def fundamental_law(ic: ArrayLike, breadth: ArrayLike) -> np.ndarray | float:
    """Grinold's information ratio :math:`IC\\sqrt{BR}`.

    Parameters
    ----------
    ic : array_like
        Information coefficient: correlation of forecasts with outcomes.
    breadth : array_like
        Independent forecasts per year.

    Returns
    -------
    float or ndarray

    Examples
    --------
    A stock picker with IC 0.05 on 400 independent bets a year:

    >>> fundamental_law(0.05, 400)
    1.0
    """
    return _scalar(np.asarray(ic, dtype=np.float64) * np.sqrt(np.asarray(breadth, dtype=np.float64)))


def simulate_active_management(ic: float, breadth: int, n_periods: int, rng: np.random.Generator | int | None = None) -> ActiveManagementResult:
    """Simulate a manager with skill ``ic`` making ``breadth`` independent bets a period.

    Each period the standardized residual returns are :math:`z_i \\sim N(0, 1)`
    and the manager's scores are :math:`g_i = IC\\, z_i + \\sqrt{1 - IC^2}\\,\\varepsilon_i`;
    the optimal active positions are proportional to the scores, so the
    active return is :math:`\\sum_i g_i z_i`. Its exact information ratio
    is :math:`IC\\sqrt{BR}/\\sqrt{1 + IC^2}`, which Grinold's law
    approximates to first order in :math:`IC`.

    Parameters
    ----------
    ic : float
        Information coefficient in ``(-1, 1)``.
    breadth : int
        Independent bets per period.
    n_periods : int
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    ActiveManagementResult
    """
    gen = np.random.default_rng(rng)
    z = gen.standard_normal((n_periods, breadth))
    g = ic * z + np.sqrt(1.0 - ic**2) * gen.standard_normal((n_periods, breadth))
    active = (g * z).sum(axis=1) / np.sqrt(breadth)
    ir = float(active.mean() / active.std(ddof=1))
    zc, gc = z - z.mean(axis=1, keepdims=True), g - g.mean(axis=1, keepdims=True)
    ics = (zc * gc).sum(axis=1) / np.sqrt((zc**2).sum(axis=1) * (gc**2).sum(axis=1))
    return ActiveManagementResult(
        active_returns=active,
        ir=ir,
        ir_se=float(np.sqrt((1.0 + 0.5 * ir**2) / n_periods)),
        predicted_ir=float(ic * np.sqrt(breadth)),
        realized_ic=float(ics.mean()),
        breadth=int(breadth),
    )
