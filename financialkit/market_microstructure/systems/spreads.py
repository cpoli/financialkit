"""Bid-ask spreads: Demsetz's cost of immediacy and two estimators that need no quotes.

Harold Demsetz (1968) called the spread the price a dealer charges for
*immediacy* and showed that it falls with trading activity. Quoted,
effective and realized spreads measure that price from quotes and
trades; the realized spread is what the liquidity provider keeps and the
price impact is what it loses to informed traders (Huang and Stoll,
1996). When there are no quotes, Richard Roll (1984) read the spread off
the negative autocovariance of price changes that the bid-ask bounce
causes, and Shane Corwin and Paul Schultz (2012) off daily highs and
lows, which are usually at the ask and the bid.

References: H. Demsetz, "The Cost of Transacting," *Quarterly J.
Economics* 82 (1968) 33-53; R. Roll, "A Simple Implicit Measure of the
Effective Bid-Ask Spread in an Efficient Market," *J. Finance* 39 (1984)
1127-1139; R. D. Huang and H. R. Stoll, "Dealer versus Auction
Markets," *J. Financial Economics* 41 (1996) 313-357; S. A. Corwin and
P. Schultz, "A Simple Way to Estimate Bid-Ask Spreads from Daily High
and Low Prices," *J. Finance* 67 (2012) 719-760.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.market_microstructure.utils.quotes import midquote, trade_signs
from financialkit.time_series.systems.efficiency import autocovariance

__all__ = ["quoted_spread", "effective_spread", "realized_spread", "price_impact", "roll_spread", "corwin_schultz_spread"]


def quoted_spread(bid: ArrayLike, ask: ArrayLike, relative: bool = False) -> np.ndarray:
    """Quoted spread :math:`a - b`, or :math:`(a - b) / m` if `relative`.

    Examples
    --------
    >>> quoted_spread([99.95], [100.05], relative=True).round(6).tolist()
    [0.001]
    """
    s = np.asarray(ask, dtype=np.float64) - np.asarray(bid, dtype=np.float64)
    return s / midquote(bid, ask) if relative else s


def effective_spread(price: ArrayLike, bid: ArrayLike, ask: ArrayLike, side: ArrayLike, relative: bool = False) -> np.ndarray:
    """Effective spread :math:`2 q (p - m)`: twice the premium paid over the midquote.

    Parameters
    ----------
    price : array_like, shape (n,)
        Trade prices.
    bid, ask : array_like, shape (n,)
        Quotes prevailing at each trade.
    side : array_like, shape (n,)
        ``+1`` for a buyer-initiated trade, ``-1`` for a seller-initiated one.
    relative : bool
        Divide by the midquote.

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    A buy inside the quotes pays less than the quoted spread.

    >>> effective_spread([100.03], [99.95], [100.05], [1]).round(6).tolist()
    [0.06]
    """
    m = midquote(bid, ask)
    s = 2.0 * trade_signs(side) * (np.asarray(price, dtype=np.float64) - m)
    return s / m if relative else s


def realized_spread(price: ArrayLike, later_mid: ArrayLike, side: ArrayLike) -> np.ndarray:
    """Realized spread :math:`2 q (p - m_{t+\\Delta})`: what the liquidity provider keeps.

    Parameters
    ----------
    price : array_like, shape (n,)
    later_mid : array_like, shape (n,)
        Midquote a fixed interval (five minutes, say) after each trade.
    side : array_like, shape (n,)

    Returns
    -------
    ndarray, shape (n,)
    """
    return 2.0 * trade_signs(side) * (np.asarray(price, dtype=np.float64) - np.asarray(later_mid, dtype=np.float64))


def price_impact(bid: ArrayLike, ask: ArrayLike, later_mid: ArrayLike, side: ArrayLike) -> np.ndarray:
    """Price impact :math:`2 q (m_{t+\\Delta} - m_t)`, so effective = realized + impact.

    Examples
    --------
    >>> p, b, a, later, q = [100.05], [99.95], [100.05], [100.02], [1]
    >>> float(effective_spread(p, b, a, q)[0] - realized_spread(p, later, q)[0] - price_impact(b, a, later, q)[0]) == 0.0
    True
    """
    return 2.0 * trade_signs(side) * (np.asarray(later_mid, dtype=np.float64) - midquote(bid, ask))


def roll_spread(prices: ArrayLike) -> float:
    """Roll's estimator :math:`s = 2\\sqrt{-\\operatorname{cov}(\\Delta p_t, \\Delta p_{t-1})}`.

    If trades alternate at random between bid and ask around an efficient
    price, the bounce makes the first-order autocovariance of price
    changes :math:`-s^2/4`.

    Parameters
    ----------
    prices : array_like, shape (n,)
        Transaction prices (or log prices, giving a relative spread).

    Returns
    -------
    float
        ``nan`` when the autocovariance is positive, which happens in
        practice and which the model cannot explain.

    Examples
    --------
    Bounce of 0.1 either side of a constant value:

    >>> import numpy as np
    >>> p = 100.0 + 0.1 * np.random.default_rng(0).choice([-1.0, 1.0], 100_000)
    >>> round(roll_spread(p), 2)
    0.2
    """
    dp = np.diff(np.asarray(prices, dtype=np.float64))
    if dp.size < 3:
        raise ValueError("need at least four prices")
    c1 = autocovariance(dp, 1)[1]
    return float(2.0 * np.sqrt(-c1)) if c1 < 0 else float("nan")


def corwin_schultz_spread(high: ArrayLike, low: ArrayLike, clip: bool = True) -> np.ndarray:
    """Corwin-Schultz relative spread from each pair of consecutive days' highs and lows.

    The variance in a day's high-low range grows with the horizon but the
    spread does not, so comparing one two-day range with two one-day
    ranges separates them:

    .. math::

       \\alpha = \\frac{\\sqrt{2\\beta} - \\sqrt\\beta}{3 - 2\\sqrt2} - \\sqrt{\\frac{\\gamma}{3 - 2\\sqrt2}}, \\qquad
       S = \\frac{2(e^\\alpha - 1)}{1 + e^\\alpha},

    with :math:`\\beta` the sum of the two squared daily log ranges and
    :math:`\\gamma` the squared two-day log range.

    Parameters
    ----------
    high, low : array_like, shape (n,)
        Daily highs and lows.
    clip : bool
        Set negative two-day estimates to zero, as Corwin and Schultz do
        before averaging.

    Returns
    -------
    ndarray, shape (n - 1,)
        Average over a month for a monthly estimate.

    Examples
    --------
    >>> corwin_schultz_spread([101.0, 101.2], [99.0, 99.1]).round(4).tolist()
    [0.0169]
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    if np.any(h < l) or np.any(l <= 0):
        raise ValueError("need 0 < low <= high")
    r2 = np.log(h / l) ** 2
    beta = r2[:-1] + r2[1:]
    gamma = np.log(np.maximum(h[:-1], h[1:]) / np.minimum(l[:-1], l[1:])) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * np.expm1(alpha) / (1.0 + np.exp(alpha))
    return np.maximum(s, 0.0) if clip else s
