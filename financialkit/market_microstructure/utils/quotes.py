"""Quote arithmetic shared by the spread, classification and impact estimators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["midquote", "align_quotes", "trade_signs"]


def midquote(bid: ArrayLike, ask: ArrayLike) -> np.ndarray:
    """The quote midpoint :math:`(b + a) / 2`, checking the quotes are not crossed.

    Examples
    --------
    >>> midquote([99.9, 100.0], [100.1, 100.4]).tolist()
    [100.0, 100.2]
    """
    b = np.asarray(bid, dtype=np.float64)
    a = np.asarray(ask, dtype=np.float64)
    if np.any(a < b):
        raise ValueError("ask must not be below bid")
    return 0.5 * (b + a)


def align_quotes(trade_times: ArrayLike, quote_times: ArrayLike, bid: ArrayLike, ask: ArrayLike, delay: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The quotes in force `delay` time units before each trade.

    Lee and Ready (1991) compared each trade with the quote standing five
    seconds earlier, because quote updates triggered by a trade were
    often time-stamped before it.

    Parameters
    ----------
    trade_times : array_like, shape (n,)
    quote_times : array_like, shape (m,)
        Increasing times at which the quote changed.
    bid, ask : array_like, shape (m,)
        Quote after each change.
    delay : float
        Look-back, in the units of the times.

    Returns
    -------
    bid, ask : ndarray, shape (n,)
        ``nan`` where no quote had yet been posted.

    Examples
    --------
    >>> b, a = align_quotes([3.0, 7.0], [0.0, 5.0], [10.0, 11.0], [10.2, 11.2], delay=2.5)
    >>> b.tolist(), a.tolist()
    ([10.0, 10.0], [10.2, 10.2])
    """
    qt = np.asarray(quote_times, dtype=np.float64)
    if np.any(np.diff(qt) < 0):
        raise ValueError("quote_times must be increasing")
    idx = np.searchsorted(qt, np.asarray(trade_times, dtype=np.float64) - delay, side="right") - 1
    ok = idx >= 0
    out_b = np.full(idx.shape, np.nan)
    out_a = np.full(idx.shape, np.nan)
    out_b[ok] = np.asarray(bid, dtype=np.float64)[idx[ok]]
    out_a[ok] = np.asarray(ask, dtype=np.float64)[idx[ok]]
    return out_b, out_a


def trade_signs(side: ArrayLike) -> np.ndarray:
    """Validate trade signs: every entry must be ``+1`` (buy) or ``-1`` (sell).

    Examples
    --------
    >>> trade_signs([1, -1, 1]).tolist()
    [1.0, -1.0, 1.0]
    """
    q = np.asarray(side, dtype=np.float64)
    if not np.all(np.abs(q) == 1.0):
        raise ValueError("trade signs must be +1 or -1")
    return q
