"""Characteristic-sorted portfolios and long-short factors.

Fama and French (1993) and Carhart (1997) build their factors from the
cross-section of stocks: each period, sort on a characteristic known at
the start of the period (size, book-to-market, past return), and hold
the top group long and the bottom group short. Fama and French build
each factor from independent 2 x 3 sorts on size and the characteristic. The same sorts give the
test portfolios that a factor model must price.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["as_panel", "sort_portfolios", "long_short_factor", "momentum_signal", "two_by_three_portfolios"]


def as_panel(x: ArrayLike) -> np.ndarray:
    """A float array of shape ``(T, n)``; a vector becomes a single column.

    Examples
    --------
    >>> as_panel([1.0, 2.0]).shape
    (2, 1)
    """
    a = np.asarray(x, dtype=np.float64)
    return a[:, None] if a.ndim == 1 else a


def _groups(signal: np.ndarray, n_groups: int) -> np.ndarray:
    """Group index ``0 .. n_groups - 1`` of each asset by the rank of ``signal`` in its row (``-1`` where nan)."""
    out = np.full(signal.shape, -1, dtype=np.int64)
    for t, row in enumerate(signal):
        ok = np.isfinite(row)
        k = int(ok.sum())
        if k >= n_groups:
            ranks = np.argsort(np.argsort(row[ok]))
            out[t, ok] = ranks * n_groups // k
    return out


def sort_portfolios(signal: ArrayLike, returns: ArrayLike, n_groups: int = 10) -> np.ndarray:
    """Equal-weighted returns of portfolios formed on ``signal``, lowest group first.

    Parameters
    ----------
    signal : array_like, shape (T, n) or (n,)
        Sorting characteristic, known when the period's returns begin; a
        vector is a characteristic that does not change. ``nan`` drops an
        asset from that period's sort.
    returns : array_like, shape (T, n)
    n_groups : int

    Returns
    -------
    ndarray, shape (T, n_groups)
        ``nan`` in periods with fewer valid assets than groups.

    Examples
    --------
    >>> sort_portfolios([3.0, 1.0, 2.0, 4.0], [[0.3, 0.1, 0.2, 0.4]], n_groups=2).tolist()
    [[0.15000000000000002, 0.35]]
    """
    R = np.atleast_2d(np.asarray(returns, dtype=np.float64))
    S = np.broadcast_to(np.asarray(signal, dtype=np.float64), R.shape)
    g = _groups(S, n_groups)
    out = np.full((R.shape[0], n_groups), np.nan)
    for j in range(n_groups):
        mask = g == j
        cnt = mask.sum(axis=1)
        with np.errstate(invalid="ignore"):
            out[:, j] = np.where(cnt > 0, np.where(mask, R, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    return out


def long_short_factor(signal: ArrayLike, returns: ArrayLike, quantile: float = 0.3) -> np.ndarray:
    """Return of the top ``quantile`` of assets by ``signal`` minus the bottom ``quantile``.

    Parameters
    ----------
    signal : array_like, shape (T, n) or (n,)
        High values go long: pass minus the size to get SMB, the
        book-to-market ratio to get HML, the past return to get UMD.
    returns : array_like, shape (T, n)
    quantile : float
        Fraction of assets in each leg, in ``(0, 0.5]`` (Fama and French
        use 30%).

    Returns
    -------
    ndarray, shape (T,)

    Examples
    --------
    >>> round(float(long_short_factor([1.0, 2.0, 3.0, 4.0], [[0.01, 0.02, 0.03, 0.05]], 0.5)[0]), 12)
    0.025
    """
    if not 0.0 < quantile <= 0.5:
        raise ValueError(f"quantile must lie in (0, 0.5], got {quantile}")
    R = np.atleast_2d(np.asarray(returns, dtype=np.float64))
    S = np.broadcast_to(np.asarray(signal, dtype=np.float64), R.shape)
    out = np.full(R.shape[0], np.nan)
    for t in range(R.shape[0]):
        ok = np.isfinite(S[t])
        k = int(np.floor(quantile * ok.sum()))
        if k == 0:
            continue
        order = np.argsort(S[t, ok])
        r = R[t, ok]
        out[t] = r[order[-k:]].mean() - r[order[:k]].mean()
    return out


def momentum_signal(returns: ArrayLike, lookback: int = 12, skip: int = 1) -> np.ndarray:
    """Jegadeesh-Titman momentum: the compound return over months ``t - lookback`` to ``t - skip - 1``.

    Parameters
    ----------
    returns : array_like, shape (T, n)
        Simple returns.
    lookback : int
        Length of the formation window, including the skipped months.
    skip : int
        Most recent months left out (one month avoids short-term reversal).

    Returns
    -------
    ndarray, shape (T, n)
        Signal available at the start of each period; the first
        ``lookback`` rows are ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> momentum_signal(np.full((4, 1), 0.1), lookback=3, skip=1)[3].round(12).tolist()
    [0.21]
    """
    R = as_panel(returns)
    logw = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(np.log1p(R), axis=0)])
    out = np.full(R.shape, np.nan)
    t = np.arange(lookback, R.shape[0])
    out[t] = np.expm1(logw[t - skip] - logw[t - lookback])
    return out


def two_by_three_portfolios(size: ArrayLike, signal: ArrayLike, returns: ArrayLike) -> np.ndarray:
    """Equal-weighted returns of Fama and French's six size and characteristic portfolios.

    Each period, assets split at the median ``size`` into small and big,
    and independently at the 30th and 70th percentiles of ``signal`` into
    low, middle and high.

    Parameters
    ----------
    size : array_like, shape (T, n) or (n,)
        Market capitalisation (or its log) at the start of each period.
    signal : array_like, shape (T, n) or (n,)
        Second characteristic, such as book-to-market.
    returns : array_like, shape (T, n)

    Returns
    -------
    ndarray, shape (T, 2, 3)
        ``[:, 0]`` small and ``[:, 1]`` big; ``[..., 0]`` low, ``[..., 2]`` high.
        ``nan`` where a portfolio is empty.

    Examples
    --------
    >>> six = two_by_three_portfolios([1, 2, 3, 4, 5, 6], [1, 3, 6, 2, 4, 5], [[0.1, 0.3, 0.6, 0.2, 0.4, 0.5]])
    >>> six[0].round(12).tolist()
    [[0.1, 0.3, 0.6], [0.2, 0.4, 0.5]]
    """
    R = np.atleast_2d(np.asarray(returns, dtype=np.float64))
    Z = np.broadcast_to(np.asarray(size, dtype=np.float64), R.shape)
    S = np.broadcast_to(np.asarray(signal, dtype=np.float64), R.shape)
    ok = np.isfinite(Z) & np.isfinite(S) & np.isfinite(R)
    with np.errstate(invalid="ignore"):
        big = Z > np.nanmedian(np.where(ok, Z, np.nan), axis=1, keepdims=True)
        lo, hi = np.nanquantile(np.where(ok, S, np.nan), [0.3, 0.7], axis=1, keepdims=True)
    terciles = [S <= lo, (S > lo) & (S <= hi), S > hi]
    out = np.full((R.shape[0], 2, 3), np.nan)
    for i, row in enumerate([~big, big]):
        for j, col in enumerate(terciles):
            mask = ok & row & col
            cnt = mask.sum(axis=1)
            out[:, i, j] = np.where(cnt > 0, np.where(mask, R, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    return out
