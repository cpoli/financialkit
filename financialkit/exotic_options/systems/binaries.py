"""Binary (digital) options: cash-or-nothing, asset-or-nothing, gap and one-touch.

Reiner and Rubinstein (1991) "unscrambled the binary code": every
vanilla and barrier option decomposes into a handful of binary
building blocks, and each block has a closed form.

* :func:`cash_or_nothing_price` -- pays a fixed cash amount if the option
  finishes in the money: :math:`X e^{-rT} N(\\pm d_2)`.
* :func:`asset_or_nothing_price` -- pays the asset itself:
  :math:`S e^{-qT} N(\\pm d_1)`. A vanilla call is an asset-or-nothing
  call minus :math:`K` cash-or-nothing calls.
* :func:`gap_option_price` -- pays :math:`S_T - K_2` when
  :math:`S_T > K_1`, and can be worth less than zero.
* :func:`one_touch_price` -- pays cash the first time the price touches a
  barrier: the rebate term of :func:`~financialkit.exotic_options.systems.barriers.barrier_price`
  on its own.

References: M. Rubinstein and E. Reiner, "Unscrambling the Binary Code,"
*Risk* 4(9) (1991) 75-83; E. G. Haug, *The Complete Guide to Option
Pricing Formulas*, 2nd ed. (McGraw-Hill, 2007), Sections 4.19.1-4.19.3;
Hull, *Options, Futures, and Other Derivatives*, 10th ed., Section 26.10.

At ``T = 0`` or ``sigma = 0`` each price is its limit, with
:math:`N(d_{1,2})` the probability of exercise, 1 or 0, and 1/2 exactly at
the money (see :func:`~financialkit.options.black_d1_d2`). Negative
``S``, ``K``, ``T`` or ``sigma`` raise :class:`ValueError`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from financialkit._validation import nonnegative, positive
from financialkit._validation import option_type as check_option_type
from financialkit.exotic_options.utils.numerics import deterministic_hit_time
from financialkit.options import black_d1_d2
from financialkit.options.utils.common import as_output

__all__ = ["cash_or_nothing_price", "asset_or_nothing_price", "gap_option_price", "one_touch_price"]


def _d1_d2(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike) -> tuple[np.ndarray, ...]:
    S, K, T, r, sigma, q = (np.asarray(a, dtype=np.float64) for a in (S, K, T, r, sigma, q))
    nonnegative(S=S, K=K, T=T, sigma=sigma)
    return S, K, T, r, q, *black_d1_d2(S * np.exp((r - q) * T), K, sigma, T)


def cash_or_nothing_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0, option_type: str = "call", cash: ArrayLike = 1.0
) -> np.ndarray | float:
    """Cash-or-nothing digital: `cash` at expiry if :math:`S_T > K` (call) or :math:`S_T < K` (put).

    .. math:: c = X e^{-rT} N(d_2), \\qquad p = X e^{-rT} N(-d_2).

    Parameters
    ----------
    S, K, T, r, sigma : array_like
        Spot, strike, maturity, rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}
    cash : array_like
        Amount paid :math:`X`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Haug (2007), Section 4.19.2: S = 100, K = 80, T = 0.75, r = 6%,
    b = 0, sigma = 35%, cash 10:

    >>> round(cash_or_nothing_price(100.0, 80.0, 0.75, 0.06, 0.35, q=0.06, option_type="put", cash=10.0), 3)
    2.671
    """
    phi = check_option_type(option_type)
    _, _, T, r, _, _, d2 = _d1_d2(S, K, T, r, sigma, q)
    return as_output(np.asarray(cash, dtype=np.float64) * np.exp(-r * T) * norm.cdf(phi * d2))


def asset_or_nothing_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0, option_type: str = "call"
) -> np.ndarray | float:
    """Asset-or-nothing digital: one share at expiry if the option finishes in the money.

    .. math:: c = S e^{-qT} N(d_1), \\qquad p = S e^{-qT} N(-d_1).

    Parameters
    ----------
    S, K, T, r, sigma : array_like
        Spot, strike, maturity, rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    Haug (2007), Section 4.19.3: S = 70, K = 65, T = 0.5, r = 7%, b = 2%,
    sigma = 27%:

    >>> round(asset_or_nothing_price(70.0, 65.0, 0.5, 0.07, 0.27, q=0.05, option_type="put"), 4)
    20.2069
    """
    phi = check_option_type(option_type)
    S, _, T, _, q, d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return as_output(S * np.exp(-q * T) * norm.cdf(phi * d1))


def gap_option_price(
    S: ArrayLike, K1: ArrayLike, K2: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0, option_type: str = "call"
) -> np.ndarray | float:
    """Gap option: pays :math:`S_T - K_2` if :math:`S_T > K_1` (call), :math:`K_2 - S_T` if :math:`S_T < K_1` (put).

    The call is an asset-or-nothing call minus :math:`K_2` cash-or-nothing
    calls, and the put :math:`K_2` cash-or-nothing puts minus an
    asset-or-nothing put, all triggered at :math:`K_1`:

    .. math:: c = S e^{-qT} N(d_1) - K_2 e^{-rT} N(d_2), \\quad
              p = K_2 e^{-rT} N(-d_2) - S e^{-qT} N(-d_1), \\quad d_{1,2} \\text{ at } K_1.

    Parameters
    ----------
    S : array_like
        Spot.
    K1, K2 : array_like
        Trigger and payoff strikes.
    T, r, sigma : array_like
        Maturity, rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    Haug (2007), Section 4.19.1: S = 50, K1 = 50, K2 = 57, T = 0.5, r = 9%,
    b = 9%, sigma = 20%; the value is negative:

    >>> round(gap_option_price(50.0, 50.0, 57.0, 0.5, 0.09, 0.20), 4)
    -0.0053
    """
    phi = check_option_type(option_type)
    return as_output(
        phi
        * (
            np.asarray(asset_or_nothing_price(S, K1, T, r, sigma, q, option_type))
            - np.asarray(cash_or_nothing_price(S, K1, T, r, sigma, q, option_type, cash=K2))
        )
    )


def one_touch_price(S: ArrayLike, H: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0, cash: ArrayLike = 1.0) -> np.ndarray | float:
    """One-touch: `cash` paid the first time the price touches :math:`H` before :math:`T`.

    The barrier is below the spot for :math:`H < S` and above it
    otherwise. With :math:`\\mu = (b - \\sigma^2/2)/\\sigma^2`,
    :math:`\\lambda = \\sqrt{\\mu^2 + 2r/\\sigma^2}` and :math:`\\eta = \\pm 1`
    for a down or up barrier,

    .. math:: V = X\\left[\\left(\\tfrac{H}{S}\\right)^{\\mu + \\lambda} N(\\eta z)
              + \\left(\\tfrac{H}{S}\\right)^{\\mu - \\lambda} N(\\eta z - 2\\eta\\lambda\\sigma\\sqrt T)\\right],
              \\quad z = \\frac{\\ln(H/S)}{\\sigma\\sqrt T} + \\lambda\\sigma\\sqrt T,

    the rebate term of a knock-out barrier option (Reiner and Rubinstein
    1991). As :math:`T \\to \\infty` it tends to the perpetual value
    :math:`X (H/S)^{\\mu + \\eta\\lambda}`. With ``sigma = 0``, at ``T = 0``
    or at ``S = 0`` the path :math:`S e^{bt}` is known, and the price is
    `cash` discounted from the date it reaches :math:`H`, or 0 if it never does.

    Parameters
    ----------
    S, H, T, r, sigma : array_like
        Spot, barrier, maturity, rate and volatility.
    q : array_like
        Continuous dividend yield.
    cash : array_like
        Amount paid at the touch.

    Returns
    -------
    float or ndarray

    Examples
    --------
    A barrier at the spot is touched at once:

    >>> one_touch_price(100.0, 100.0, 1.0, 0.05, 0.2)
    1.0
    """
    S, H, T, r, sigma, q, X = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (S, H, T, r, sigma, q, cash)))
    nonnegative(S=S, T=T, sigma=sigma)
    positive(H=H)
    is_down = H < S
    eta = np.where(is_down, 1.0, -1.0)
    v = sigma * np.sqrt(T)
    regular = (v > 0) & (S > 0)
    Ss, sig, TT = np.where(regular, S, H), np.where(regular, sigma, 1.0), np.where(regular, T, 1.0)
    vv = sig * np.sqrt(TT)
    mu = (r - q - 0.5 * sig**2) / sig**2
    lam = np.sqrt(mu**2 + 2.0 * r / sig**2)
    hs = H / Ss
    z = np.log(hs) / vv + lam * vv
    value = X * (hs ** (mu + lam) * norm.cdf(eta * z) + hs ** (mu - lam) * norm.cdf(eta * z - 2.0 * eta * lam * vv))
    hit, t_hit = deterministic_hit_time(S, H, T, r - q, is_down)
    deterministic = np.where(hit, X * np.exp(-r * t_hit), 0.0)
    return as_output(np.where(H == S, X, np.where(regular, value, deterministic)))
