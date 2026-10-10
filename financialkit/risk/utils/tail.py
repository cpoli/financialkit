"""Tail numerics shared by the VaR, expected-shortfall and backtesting modules.

:func:`weighted_tail` is the one quantile-and-tail-mean rule every
scenario-based estimator uses. With equal weights it is Acerbi and
Tasche's (2002) estimator: the VaR is the order statistic
:math:`-X_{(\\lceil n p \\rceil)}` and the ES averages the
:math:`\\lfloor n p \\rfloor` worst outcomes plus the fraction of the next
one needed to fill a tail of probability exactly :math:`p = 1 - \\alpha`.

References: C. Acerbi and D. Tasche, *J. Banking Finance* 26 (2002)
1487-1503.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import rankdata

__all__ = ["check_alpha", "weighted_tail", "var_exceptions", "pseudo_observations"]


def check_alpha(alpha: float) -> float:
    """Validate a confidence level and return it as a float.

    Parameters
    ----------
    alpha : float
        Confidence level, strictly between 0 and 1.

    Returns
    -------
    float
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    return float(alpha)


def weighted_tail(pnl: ArrayLike, alpha: float, weights: ArrayLike | None = None) -> tuple[float, float]:
    """VaR and expected shortfall of a discrete profit-and-loss distribution.

    Parameters
    ----------
    pnl : array_like, shape (n,)
        Scenario profits and losses (losses negative).
    alpha : float
        Confidence level.
    weights : array_like, shape (n,), optional
        Scenario probabilities (normalised here); equal if omitted.

    Returns
    -------
    var, es : float
        Positive losses. ``es >= var`` always.

    Examples
    --------
    Ten equally likely outcomes, a 15% tail: the VaR is the second-worst
    loss and the ES averages the worst loss with half of the second.

    >>> weighted_tail([-10, -6, -4, -1, 0, 1, 2, 3, 4, 5], 0.85)
    (6.0, 8.666666666666666)
    """
    p = 1.0 - check_alpha(alpha)
    x = np.asarray(pnl, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("pnl is empty")
    w = np.full(x.size, 1.0 / x.size) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != x.shape or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, not all zero, and match pnl")
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order] / w.sum()
    cum = np.cumsum(w)
    # A tiny slack keeps k = n p exact when the cumulative sum rounds below it.
    k = min(int(np.searchsorted(cum, p - 1e-12 * p, side="left")), x.size - 1)
    below = cum[k] - w[k]
    es = -(np.dot(w[:k], x[:k]) + (p - below) * x[k]) / p
    return float(-x[k]), float(es)


def var_exceptions(pnl: ArrayLike, var: ArrayLike) -> np.ndarray:
    """Days on which the loss exceeded the VaR.

    Parameters
    ----------
    pnl : array_like, shape (n,)
        Realised profits and losses.
    var : array_like, shape (n,) or scalar
        VaR forecast for each day, a positive loss.

    Returns
    -------
    ndarray of bool, shape (n,)
        The hit sequence: ``pnl < -var``.
    """
    return np.asarray(pnl, dtype=np.float64) < -np.asarray(var, dtype=np.float64)


def pseudo_observations(x: ArrayLike) -> np.ndarray:
    """Column-wise ranks scaled into the open unit interval, ``rank / (n + 1)``.

    The empirical copula of the data: it discards the marginals and keeps
    only the dependence, as Sklar's theorem says one can.

    Parameters
    ----------
    x : array_like, shape (n, d)

    Returns
    -------
    ndarray, shape (n, d)
    """
    x = np.asarray(x, dtype=np.float64)
    return rankdata(x, axis=0) / (x.shape[0] + 1.0)
