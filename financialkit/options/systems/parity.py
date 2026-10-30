"""Model-free relations between European option prices: put-call parity
and Merton's no-arbitrage bounds.

Both follow from static replication alone, with no model for the
underlying. A call minus a put is a forward,
:math:`c - p = S e^{-qT} - K e^{-rT}` (Stoll 1969), and a call is worth
at least the forward's value and at most the stock,
:math:`\\max(S e^{-qT} - K e^{-rT}, 0) \\le c \\le S e^{-qT}` (Merton 1973).
:func:`~financialkit.options.systems.implied_vol.implied_volatility`
checks these bounds before it searches for a volatility.

References: H. R. Stoll, *J. Finance* 24 (1969) 801-824; R. C. Merton,
*Bell J. Econ. Manag. Sci.* 4 (1973) 141-183; Hull, *Options, Futures,
and Other Derivatives*, 10th ed., Ch. 11.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit._validation import option_type as check_option_type
from financialkit.options.utils.common import as_output

__all__ = ["put_call_parity", "parity_residual", "no_arbitrage_bounds"]


def _pv_forward(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike) -> np.ndarray:
    S, K, T, r, q = (np.asarray(a, dtype=np.float64) for a in (S, K, T, r, q))
    return S * np.exp(-q * T) - K * np.exp(-r * T)


def put_call_parity(
    price: ArrayLike, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike = 0.0, option_type: str = "call"
) -> np.ndarray | float:
    """The other leg of a European call-put pair, by put-call parity.

    Parameters
    ----------
    price : array_like
        Price of the known option.
    S, K, T, r, q : array_like
        Spot, strike, maturity, risk-free rate and dividend yield.
    option_type : {"call", "put"}
        Type of the *known* option; the other type's price is returned.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Hull, 10th ed., Example 15.6: a 4.76 call with S = 42, K = 40,
    r = 10% and six months implies a 0.81 put.

    >>> round(put_call_parity(4.7594, 42.0, 40.0, 0.5, 0.10), 2)
    0.81
    """
    return as_output(np.asarray(price, dtype=np.float64) - check_option_type(option_type) * _pv_forward(S, K, T, r, q))


def parity_residual(call: ArrayLike, put: ArrayLike, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike = 0.0) -> np.ndarray | float:
    """Violation of put-call parity, :math:`c - p - (S e^{-qT} - K e^{-rT})`.

    Zero for any arbitrage-free pair; a nonzero residual is the profit of
    the conversion (or reversal) that exploits it.

    Parameters
    ----------
    call, put : array_like
        European call and put prices with the same strike and maturity.
    S, K, T, r, q : array_like

    Returns
    -------
    float or ndarray
    """
    return as_output(np.asarray(call, dtype=np.float64) - np.asarray(put, dtype=np.float64) - _pv_forward(S, K, T, r, q))


def no_arbitrage_bounds(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike = 0.0, option_type: str = "call"
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Merton's (1973) model-free bounds on a European option price.

    Calls: :math:`\\max(S e^{-qT} - K e^{-rT}, 0) \\le c \\le S e^{-qT}`.
    Puts: :math:`\\max(K e^{-rT} - S e^{-qT}, 0) \\le p \\le K e^{-rT}`.

    Parameters
    ----------
    S, K, T, r, q : array_like
    option_type : {"call", "put"}

    Returns
    -------
    lower, upper : float or ndarray

    Examples
    --------
    >>> lo, hi = no_arbitrage_bounds(100.0, 90.0, 1.0, 0.05)
    >>> round(lo, 4), hi
    (14.3894, 100.0)
    """
    sign = check_option_type(option_type)
    S, K, T, r, q = (np.asarray(a, dtype=np.float64) for a in (S, K, T, r, q))
    lower = np.maximum(sign * _pv_forward(S, K, T, r, q), 0.0)
    upper = S * np.exp(-q * T) if sign == 1 else K * np.exp(-r * T)
    lower, upper = np.broadcast_arrays(lower, upper)
    return as_output(lower), as_output(upper)
