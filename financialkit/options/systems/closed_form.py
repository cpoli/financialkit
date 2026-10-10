"""Closed-form European option prices: Bachelier, Black-Scholes-Merton,
Black-76 and Garman-Kohlhagen.

All four are one formula seen through different underlyings. Each prices
off the forward :math:`F = S e^{(r - q) T}` and discounts at :math:`r`:
Black-Scholes-Merton is Black-76 on the forward of a dividend-paying
stock, and Garman-Kohlhagen is Black-Scholes-Merton with the foreign
interest rate as the dividend yield. Bachelier's model is the arithmetic
(normal) counterpart, with an absolute volatility in price units. Every
pricer broadcasts over its array inputs.

References: L. Bachelier, *Théorie de la spéculation* (1900); F. Black and
M. Scholes, *J. Political Economy* 81 (1973) 637-654; R. C. Merton, *Bell
J. Econ. Manag. Sci.* 4 (1973) 141-183; F. Black, *J. Financial Economics*
3 (1976) 167-179; M. B. Garman and S. W. Kohlhagen, *J. Int. Money and
Finance* 2 (1983) 231-237; Hull, *Options, Futures, and Other
Derivatives*, 10th ed., Chs. 15, 17-18.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from financialkit.options.utils.common import as_output, option_sign

__all__ = ["bachelier_price", "black_scholes_price", "black76_price", "garman_kohlhagen_price"]


def _black_undiscounted(F: np.ndarray, K: np.ndarray, sigma: np.ndarray, T: np.ndarray, sign: int) -> np.ndarray:
    """Black's formula without discounting; intrinsic forward value where ``sigma sqrt(T) = 0``."""
    v = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * v**2) / v
        value = sign * (F * norm.cdf(sign * d1) - K * norm.cdf(sign * (d1 - v)))
    return np.where(v > 0, value, np.maximum(sign * (F - K), 0.0))


def black76_price(F: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, option_type: str = "call") -> np.ndarray | float:
    r"""Black's (1976) price of a European option on a forward or futures price.

    .. math:: c = e^{-rT}[F N(d_1) - K N(d_2)], \quad
              p = e^{-rT}[K N(-d_2) - F N(-d_1)].

    This is the single Black-76 implementation in financialkit: caps,
    floors and swaptions in :mod:`financialkit.interest_rate_models` call it.

    Parameters
    ----------
    F : array_like
        Forward (or futures) price for delivery at ``T``.
    K, T, r, sigma : array_like
        Strike, maturity (years), discount rate and lognormal volatility.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    Hull, 10th ed., Example 18.7: a 4-month put on futures at 20 with a
    20 strike, 9% rate and 25% volatility.

    >>> round(black76_price(20.0, 20.0, 4 / 12, 0.09, 0.25, "put"), 2)
    1.12
    """
    F, K, T, r, sigma = (np.asarray(a, dtype=np.float64) for a in (F, K, T, r, sigma))
    return as_output(np.exp(-r * T) * _black_undiscounted(F, K, sigma, T, option_sign(option_type)))


def black_scholes_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0, option_type: str = "call"
) -> np.ndarray | float:
    r"""Black-Scholes-Merton price of a European option on a dividend-paying stock.

    .. math:: c = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \quad
              p = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),

    with :math:`d_{1,2} = [\ln(S/K) + (r - q \pm \sigma^2/2) T] / (\sigma\sqrt{T})`.
    Black and Scholes (1973) derived it for ``q = 0``; Merton (1973) added
    the continuous dividend yield. At ``T = 0`` or ``sigma = 0`` it returns
    the discounted intrinsic value of the forward.

    Parameters
    ----------
    S, K, T, r, sigma : array_like
        Spot, strike, maturity (years), risk-free rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    Hull, 10th ed., Example 15.6 (S = 42, K = 40, r = 10%, sigma = 20%,
    six months):

    >>> round(black_scholes_price(42.0, 40.0, 0.5, 0.10, 0.20), 2)
    4.76
    >>> round(black_scholes_price(42.0, 40.0, 0.5, 0.10, 0.20, option_type="put"), 2)
    0.81
    """
    S, T, r, q = (np.asarray(a, dtype=np.float64) for a in (S, T, r, q))
    return black76_price(S * np.exp((r - q) * T), K, T, r, sigma, option_type)


def garman_kohlhagen_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r_d: ArrayLike, r_f: ArrayLike, sigma: ArrayLike, option_type: str = "call"
) -> np.ndarray | float:
    r"""Garman-Kohlhagen (1983) price of a European currency option.

    The foreign currency is an asset paying the foreign interest rate, so
    this is :func:`black_scholes_price` with ``q = r_f``:

    .. math:: c = S e^{-r_f T} N(d_1) - K e^{-r_d T} N(d_2).

    Parameters
    ----------
    S : array_like
        Spot exchange rate, domestic units per unit of foreign currency.
    K, T : array_like
        Strike (same units) and maturity in years.
    r_d, r_f : array_like
        Domestic and foreign continuously compounded rates.
    sigma : array_like
        Exchange-rate volatility.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    Haug, *The Complete Guide to Option Pricing Formulas*, 2nd ed.,
    Section 1.1.6: S = 1.56, K = 1.60, six months, r_d = 6%, r_f = 8%,
    sigma = 12%.

    >>> round(garman_kohlhagen_price(1.56, 1.60, 0.5, 0.06, 0.08, 0.12), 4)
    0.0291
    """
    return black_scholes_price(S, K, T, r_d, sigma, q=r_f, option_type=option_type)


def bachelier_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma_n: ArrayLike, q: ArrayLike = 0.0, option_type: str = "call"
) -> np.ndarray | float:
    r"""Bachelier's (1900) normal-model price of a European option.

    The forward follows an arithmetic Brownian motion with absolute
    volatility :math:`\sigma_N` (price units per square-root year), so

    .. math:: c = e^{-rT}\left[(F - K) N(d) + \sigma_N\sqrt{T}\, n(d)\right],
              \quad d = \frac{F - K}{\sigma_N \sqrt{T}},

    and the put follows from parity. At the money,
    :math:`c = e^{-rT}\sigma_N\sqrt{T/(2\pi)}`. Bachelier's own setting is
    ``r = q = 0``.

    Parameters
    ----------
    S, K, T, r : array_like
        Spot, strike, maturity (years) and risk-free rate.
    sigma_n : array_like
        Normal (absolute) volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import numpy as np
    >>> price = bachelier_price(100.0, 100.0, 1.0, 0.0, 20.0)
    >>> bool(np.isclose(price, 20.0 / np.sqrt(2 * np.pi)))
    True
    """
    sign = option_sign(option_type)
    S, K, T, r, sigma_n, q = (np.asarray(a, dtype=np.float64) for a in (S, K, T, r, sigma_n, q))
    F = S * np.exp((r - q) * T)
    v = sigma_n * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (F - K) / v
        value = sign * (F - K) * norm.cdf(sign * d) + v * norm.pdf(d)
    value = np.where(v > 0, value, np.maximum(sign * (F - K), 0.0))
    return as_output(np.exp(-r * T) * value)
