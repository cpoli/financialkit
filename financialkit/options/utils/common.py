"""Shared helpers for the options pricers: input validation, the forward,
Black's :math:`d_1, d_2`, payoffs, and finite-difference Greeks used to
check the analytic ones.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from financialkit.options.core.base import Greeks

__all__ = ["option_sign", "as_output", "forward_price", "black_d1_d2", "payoff", "finite_difference_greeks"]


def option_sign(option_type: str) -> int:
    """``+1`` for a call, ``-1`` for a put.

    Parameters
    ----------
    option_type : {"call", "put"}

    Returns
    -------
    int
    """
    if option_type == "call":
        return 1
    if option_type == "put":
        return -1
    raise ValueError(f'option_type must be "call" or "put", got {option_type!r}')


def as_output(x: ArrayLike) -> np.ndarray | float:
    """Return a Python float for a 0-d result, the array otherwise."""
    x = np.asarray(x, dtype=np.float64)
    return float(x) if x.ndim == 0 else x


def forward_price(S: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike = 0.0) -> np.ndarray | float:
    """Forward price :math:`F = S e^{(r - q) T}`.

    Parameters
    ----------
    S, T, r, q : array_like
        Spot, maturity, risk-free rate and dividend yield.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(forward_price(100.0, 1.0, 0.05), 4)
    105.1271
    """
    S, T, r, q = (np.asarray(a, dtype=np.float64) for a in (S, T, r, q))
    return as_output(S * np.exp((r - q) * T))


def black_d1_d2(F: ArrayLike, K: ArrayLike, sigma: ArrayLike, T: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    r"""Black's :math:`d_{1,2} = [\ln(F/K) \pm \tfrac12 \sigma^2 T] / (\sigma\sqrt{T})`.

    Parameters
    ----------
    F, K, sigma, T : array_like
        Forward, strike, volatility and maturity; ``sigma * sqrt(T)`` must
        be positive.

    Returns
    -------
    d1, d2 : ndarray
    """
    F, K, sigma, T = (np.asarray(a, dtype=np.float64) for a in (F, K, sigma, T))
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v**2) / v
    return d1, d1 - v


def payoff(S: ArrayLike, K: ArrayLike, option_type: str = "call") -> np.ndarray | float:
    """Exercise value :math:`\\max(\\pm(S - K), 0)`.

    Parameters
    ----------
    S, K : array_like
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray
    """
    sign = option_sign(option_type)
    return as_output(np.maximum(sign * (np.asarray(S, dtype=np.float64) - np.asarray(K, dtype=np.float64)), 0.0))


def finite_difference_greeks(
    pricer: Callable[..., ArrayLike],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    h: float = 1e-4,
) -> Greeks:
    """Greeks by central finite differences of ``pricer``.

    Used to check analytic Greeks, and to get Greeks from any pricer with
    the ``pricer(S, K, T, r, sigma, q, option_type)`` signature.

    Parameters
    ----------
    pricer : callable
        ``pricer(S, K, T, r, sigma, q, option_type) -> price``.
    S, K, T, r, sigma, q : float
    option_type : {"call", "put"}
    h : float
        Relative bump for ``S`` and absolute bump for the other inputs.

    Returns
    -------
    Greeks
    """

    def V(S=S, T=T, r=r, sigma=sigma, q=q) -> float:
        return float(np.asarray(pricer(S, K, T, r, sigma, q, option_type)))

    hS = h * S
    v0 = V()
    return Greeks(
        delta=(V(S=S + hS) - V(S=S - hS)) / (2 * hS),
        gamma=(V(S=S + hS) - 2 * v0 + V(S=S - hS)) / hS**2,
        vega=(V(sigma=sigma + h) - V(sigma=sigma - h)) / (2 * h),
        theta=-(V(T=T + h) - V(T=T - h)) / (2 * h),
        rho=(V(r=r + h) - V(r=r - h)) / (2 * h),
        epsilon=(V(q=q + h) - V(q=q - h)) / (2 * h),
        vanna=(V(S=S + hS, sigma=sigma + h) - V(S=S + hS, sigma=sigma - h) - V(S=S - hS, sigma=sigma + h) + V(S=S - hS, sigma=sigma - h)) / (4 * hS * h),
        volga=(V(sigma=sigma + h) - 2 * v0 + V(sigma=sigma - h)) / h**2,
    )
