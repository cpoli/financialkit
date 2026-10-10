"""Lee and Ready's trade classification: who initiated each trade?

Most spread, impact and informed-trading measures need the sign of each
trade, which trade records do not carry. Charles Lee and Mark Ready
(1991) combined two rules: the *quote rule* signs a trade above the
midquote a buy and below it a sell; a trade at the midquote is signed by
the *tick test*, a buy if it was above the last different price. They
also compared trades with the quote of five seconds earlier, see
:func:`~financialkit.market_microstructure.utils.quotes.align_quotes`.

Reference: C. M. C. Lee and M. J. Ready, "Inferring Trade Direction from
Intraday Data," *J. Finance* 46 (1991) 733-746.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.market_microstructure.utils.quotes import midquote

__all__ = ["tick_test", "quote_rule", "lee_ready"]


def tick_test(prices: ArrayLike) -> np.ndarray:
    """Tick test: ``+1`` on an uptick or zero-uptick, ``-1`` on a downtick or zero-downtick.

    Returns
    -------
    ndarray, shape (n,)
        ``0`` until the price first changes.

    Examples
    --------
    >>> tick_test([10.0, 10.0, 10.1, 10.1, 10.0, 10.0]).tolist()
    [0.0, 0.0, 1.0, 1.0, -1.0, -1.0]
    """
    p = np.asarray(prices, dtype=np.float64)
    tick = np.zeros_like(p)
    tick[1:] = np.sign(np.diff(p))
    # Carry the last non-zero tick forward through unchanged prices.
    idx = np.where(tick != 0, np.arange(p.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return tick[idx]


def quote_rule(prices: ArrayLike, bid: ArrayLike, ask: ArrayLike) -> np.ndarray:
    """Quote rule: ``+1`` above the midquote, ``-1`` below, ``0`` at it.

    Examples
    --------
    >>> quote_rule([10.05, 10.00, 9.96], [9.95, 9.95, 9.95], [10.05, 10.05, 10.05]).tolist()
    [1.0, 0.0, -1.0]
    """
    return np.sign(np.asarray(prices, dtype=np.float64) - midquote(bid, ask))


def lee_ready(prices: ArrayLike, bid: ArrayLike, ask: ArrayLike) -> np.ndarray:
    """Lee-Ready classification: the quote rule, with the tick test at the midquote.

    Parameters
    ----------
    prices : array_like, shape (n,)
        Trade prices in time order.
    bid, ask : array_like, shape (n,)
        Quotes to compare each trade with (already lagged if desired).

    Returns
    -------
    ndarray, shape (n,)
        ``+1`` buy, ``-1`` sell, ``0`` if neither rule decides.

    Examples
    --------
    >>> lee_ready([10.05, 10.00, 10.00, 9.95], [9.95] * 4, [10.05] * 4).tolist()
    [1.0, -1.0, -1.0, -1.0]
    """
    q = quote_rule(prices, bid, ask)
    return np.where(q != 0, q, tick_test(prices))
