"""Implied volatility: inverting Black-Scholes, its at-the-money shortcut,
and the smile across strikes.

The Black-Scholes price is strictly increasing in :math:`\\sigma`, from the
lower no-arbitrage bound at :math:`\\sigma = 0` to the upper bound as
:math:`\\sigma \\to \\infty`, so a price strictly between Merton's bounds
has exactly one implied volatility. :func:`implied_volatility` checks the
bounds first and then brackets the root for :func:`scipy.optimize.brentq`.
Latané and Rendleman (1976) were the first to read volatilities off
quoted option prices this way. Brenner and Subrahmanyam (1988) noticed
that at the money the formula is nearly linear in :math:`\\sigma`, giving
:func:`brenner_subrahmanyam_vol`. Inverting a whole strip of prices gives
the volatility smile (:func:`implied_vol_smile`), flat under
Black-Scholes and, since the 1987 crash, sharply skewed in equity markets.

References: H. A. Latané and R. J. Rendleman, *J. Finance* 31 (1976)
369-381; M. Brenner and M. G. Subrahmanyam, *Financial Analysts J.* 44(5)
(1988) 80-83; M. Rubinstein, *J. Finance* 49 (1994) 771-818; Hull,
*Options, Futures, and Other Derivatives*, 10th ed., Sections 15.11 and
20.1-20.3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.options.core.base import SmileResult
from financialkit.options.systems.closed_form import black_scholes_price
from financialkit.options.systems.parity import no_arbitrage_bounds
from financialkit.options.utils.common import as_output, option_sign

__all__ = ["implied_volatility", "brenner_subrahmanyam_vol", "implied_vol_smile"]

_SIGMA_LO = 1e-8
_SIGMA_HI = 5.0


def implied_volatility(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
    tol: float = DEFAULT_ATOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> np.ndarray | float:
    """Black-Scholes implied volatility by Brent's method.

    For the Black-76 implied volatility of an option on a forward ``F``,
    pass ``S = F`` and ``q = r``.

    Parameters
    ----------
    price : array_like
        Observed European option prices.
    S, K, T, r, q : array_like
        Spot, strike, maturity, risk-free rate and dividend yield.
    option_type : {"call", "put"}
    tol : float
        Absolute tolerance on the volatility.
    max_iter : int
        Iteration cap passed to :func:`scipy.optimize.brentq`.

    Returns
    -------
    float or ndarray

    Raises
    ------
    ValueError
        If any price lies outside Merton's open no-arbitrage interval,
        where no volatility can reproduce it.

    Examples
    --------
    Hull, 10th ed., Section 15.11: a call worth 1.875 with S = 21,
    K = 20, r = 10% and three months has an implied volatility of 23.5%.

    >>> round(implied_volatility(1.875, 21.0, 20.0, 0.25, 0.10), 3)
    0.235
    """
    option_sign(option_type)
    price, S, K, T, r, q = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (price, S, K, T, r, q)))
    lower, upper = (np.asarray(b) for b in no_arbitrage_bounds(S, K, T, r, q, option_type))
    bad = ~((price > lower) & (price < upper))
    if np.any(bad):
        i = np.flatnonzero(bad.ravel())[0]
        raise ValueError(
            f"price {price.ravel()[i]:.6g} violates the no-arbitrage bounds ({lower.ravel()[i]:.6g}, {upper.ravel()[i]:.6g}); no implied volatility exists"
        )
    vols = np.empty(price.shape)
    for idx in np.ndindex(price.shape):
        args = (S[idx], K[idx], T[idx], r[idx], q[idx])

        def f(sigma: float) -> float:
            return float(black_scholes_price(args[0], args[1], args[2], args[3], sigma, args[4], option_type)) - float(price[idx])

        hi = _SIGMA_HI
        while f(hi) < 0:
            hi *= 2.0
        vols[idx] = brentq(f, _SIGMA_LO, hi, xtol=tol, maxiter=max_iter)
    return as_output(vols)


def brenner_subrahmanyam_vol(price: ArrayLike, S: ArrayLike, T: ArrayLike, r: ArrayLike = 0.0, q: ArrayLike = 0.0) -> np.ndarray | float:
    r"""Brenner-Subrahmanyam (1988) approximation for at-the-money-forward options.

    For :math:`K = F`, :math:`c = S e^{-qT}[2N(\sigma\sqrt{T}/2) - 1]
    \approx S e^{-qT} \sigma \sqrt{T / (2\pi)}`, so

    .. math:: \sigma \approx \sqrt{\frac{2\pi}{T}} \frac{c}{S e^{-qT}},

    with a relative error of order :math:`\sigma^2 T / 24`. At the forward,
    calls and puts have the same price, so either can be passed.

    Parameters
    ----------
    price : array_like
        At-the-money-forward call (or put) price.
    S, T, r, q : array_like
        Spot, maturity, risk-free rate and dividend yield (``r`` is not used
        by the formula and is accepted for a uniform signature).

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> c = black_scholes_price(100.0, 100.0, 0.25, 0.0, 0.2)
    >>> round(brenner_subrahmanyam_vol(c, 100.0, 0.25), 4)
    0.1999
    """
    price, S, T, q = (np.asarray(a, dtype=np.float64) for a in (price, S, T, q))
    return as_output(np.sqrt(2 * np.pi / T) * price / (S * np.exp(-q * T)))


def implied_vol_smile(prices: ArrayLike, strikes: ArrayLike, S: float, T: float, r: float, q: float = 0.0, option_type: str = "call") -> SmileResult:
    """Implied volatilities across a strip of strikes at one maturity.

    Parameters
    ----------
    prices, strikes : array_like, shape (n,)
        Observed European option prices and their strikes.
    S, T, r, q : float
        Spot, maturity, risk-free rate and dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    SmileResult

    Examples
    --------
    Prices from a single Black-Scholes volatility give a flat smile:

    >>> import numpy as np
    >>> K = np.array([80.0, 100.0, 120.0])
    >>> smile = implied_vol_smile(black_scholes_price(100.0, K, 1.0, 0.03, 0.25), K, 100.0, 1.0, 0.03)
    >>> np.round(smile.vols, 6)
    array([0.25, 0.25, 0.25])
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    vols = np.asarray(implied_volatility(prices, S, strikes, T, r, q, option_type), dtype=np.float64)
    return SmileResult(strikes=strikes, vols=vols, forward=float(S * np.exp((r - q) * T)), T=float(T))
