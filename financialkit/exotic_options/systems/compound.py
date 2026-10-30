"""Options whose payoff is itself an option: compounds, forward starts and choosers.

* :func:`compound_option_price` -- Geske (1979): an option to buy or sell
  an option, priced with the bivariate normal distribution. Geske read a
  firm's equity as a call on its assets and so a stock option as a call
  on a call.
* :func:`forward_start_price` -- Rubinstein (1990): an at-the-money (or
  proportional-strike) option whose strike is set at a future date is
  worth :math:`e^{-q t}` of the option on :math:`S = 1`, times :math:`S`.
* :func:`chooser_price` -- Rubinstein (1991): choose at time :math:`t`
  whether the option is a call or a put. By put-call parity it is a
  call to :math:`T` plus a put to :math:`t`.

References: R. Geske, "The Valuation of Compound Options," *J. Financial
Economics* 7 (1979) 63-81; M. Rubinstein, "Options for the Undecided,"
*Risk* 4(4) (1991) 43; M. Rubinstein, "Exotic Options," UC Berkeley
Finance working paper 220 (1990); E. G. Haug, *The Complete Guide to Option Pricing Formulas*,
2nd ed. (McGraw-Hill, 2007), Sections 4.3, 4.4.1 and 4.6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import norm

from financialkit._validation import nonnegative, positive
from financialkit._validation import option_type as check_option_type
from financialkit.exotic_options.utils.numerics import bivariate_normal_cdf
from financialkit.options import black_scholes_price
from financialkit.options.utils.common import as_output

__all__ = ["compound_option_price", "critical_price", "forward_start_price", "chooser_price"]


def critical_price(K1: float, K2: float, tau: float, r: float, sigma: float, q: float = 0.0, underlying_type: str = "call") -> float:
    """Spot :math:`I` at which the underlying option is worth the compound strike: :math:`V(I, K_2, \\tau) = K_1`.

    Solved by :func:`scipy.optimize.brentq` on the Black-Scholes-Merton
    price of the underlying option with :math:`\\tau = T_2 - T_1` to run.

    Parameters
    ----------
    K1 : float
        Strike of the compound option (the price paid for the underlying option).
    K2 : float
        Strike of the underlying option.
    tau, r, sigma : float
        Remaining life of the underlying option at :math:`T_1`, rate and volatility.
    q : float
        Continuous dividend yield.
    underlying_type : {"call", "put"}

    Returns
    -------
    float

    Examples
    --------
    >>> from financialkit.options import black_scholes_price
    >>> I = critical_price(5.0, 100.0, 0.5, 0.05, 0.2)
    >>> round(black_scholes_price(I, 100.0, 0.5, 0.05, 0.2), 10)
    5.0
    """
    check_option_type(underlying_type, name="underlying_type")
    if underlying_type == "put" and K1 >= K2 * np.exp(-r * tau):
        raise ValueError("a put on a put needs K1 below the underlying put's maximum value K2 exp(-r tau)")

    def gap(s: float) -> float:
        return float(black_scholes_price(s, K2, tau, r, sigma, q, underlying_type)) - K1

    lo, hi = 1e-12 * K2, K2
    sign = 1.0 if underlying_type == "call" else -1.0  # the call rises with the spot, the put falls
    while sign * gap(hi) < 0:
        hi *= 2.0
    return float(brentq(gap, lo, hi, xtol=1e-14 * K2, rtol=1e-14))


def compound_option_price(
    S: ArrayLike,
    K1: float,
    K2: float,
    T1: float,
    T2: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    underlying_type: str = "call",
) -> np.ndarray | float:
    """Geske (1979) compound option: an option expiring at :math:`T_1` with strike :math:`K_1` on an option to :math:`T_2` struck at :math:`K_2`.

    With :math:`I` the critical spot (:func:`critical_price`),
    :math:`\\rho = \\sqrt{T_1/T_2}`, :math:`y_{1,2}` the Black-Scholes
    :math:`d_{1,2}` for strike :math:`I` and horizon :math:`T_1`, and
    :math:`z_{1,2}` those for :math:`K_2` and :math:`T_2`, the call on a call is

    .. math:: S e^{-q T_2} M(z_1, y_1; \\rho) - K_2 e^{-r T_2} M(z_2, y_2; \\rho) - K_1 e^{-r T_1} N(y_2),

    and the other three combinations follow by flipping signs (Haug
    2007, Sec. 4.6). They satisfy the compound put-call parity
    ``call on V - put on V = V - K1 exp(-r T1)``. At ``T1 = 0``, or with
    ``sigma = 0``, the compound option is worth its exercise value
    :math:`e^{-rT_1}\\max(\\pm(V - K_1), 0)`, with :math:`V` the underlying
    option's value at :math:`T_1` on the riskless path.

    Parameters
    ----------
    S : array_like
        Spot.
    K1, K2 : float
        Strikes of the compound option and of the underlying option.
    T1, T2 : float
        Expiries of the compound option and of the underlying, ``0 <= T1 < T2``.
    r, sigma : float
        Rate and volatility.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
        The compound option.
    underlying_type : {"call", "put"}
        The option it is written on.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Haug (2007), Section 4.6: a put on a call with S = 500, K1 = 50,
    K2 = 520, T1 = 0.25, T2 = 0.5, r = 8%, q = 3%, sigma = 35%:

    >>> round(compound_option_price(500.0, 50.0, 520.0, 0.25, 0.5, 0.08, 0.35, q=0.03, option_type="put"), 3)
    21.196
    """
    if not 0 <= T1 < T2:
        raise ValueError("need 0 <= T1 < T2")
    phi, w = check_option_type(option_type), check_option_type(underlying_type, name="underlying_type")
    S = np.asarray(S, dtype=np.float64)
    nonnegative(S=S, K1=K1, sigma=sigma)
    positive(K2=K2)
    if T1 == 0 or sigma == 0:
        underlying = np.asarray(black_scholes_price(S * np.exp((r - q) * T1), K2, T2 - T1, r, sigma, q, underlying_type))
        return as_output(np.exp(-r * T1) * np.maximum(phi * (underlying - K1), 0.0))
    I = critical_price(K1, K2, T2 - T1, r, sigma, q, underlying_type)
    v1, v2 = sigma * np.sqrt(T1), sigma * np.sqrt(T2)
    with np.errstate(divide="ignore"):  # S = 0 sends y1 and z1 to -inf, which bivariate_normal_cdf handles
        log_S = np.log(S)
    y1 = (log_S - np.log(I) + (r - q + 0.5 * sigma**2) * T1) / v1
    z1 = (log_S - np.log(K2) + (r - q + 0.5 * sigma**2) * T2) / v2
    y2, z2 = y1 - v1, z1 - v2
    rho = phi * np.sqrt(T1 / T2)
    out = phi * w * (
        S * np.exp(-q * T2) * bivariate_normal_cdf(w * z1, phi * w * y1, rho) - K2 * np.exp(-r * T2) * bivariate_normal_cdf(w * z2, phi * w * y2, rho)
    ) - phi * K1 * np.exp(-r * T1) * norm.cdf(phi * w * y2)
    return as_output(out)


def forward_start_price(
    S: ArrayLike, alpha: ArrayLike, t_start: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0, option_type: str = "call"
) -> np.ndarray | float:
    """Rubinstein (1990) forward-start option: struck at :math:`\\alpha S_{t}` at the future date :math:`t`.

    At time :math:`t` the option is worth :math:`S_t` times a
    Black-Scholes-Merton option on a unit spot with strike :math:`\\alpha`,
    and the risk-neutral value of :math:`S_t` is :math:`S e^{-qt}`:

    .. math:: V = S e^{-q t}\\, \\mathrm{BS}(1, \\alpha, T - t, r, q, \\sigma).

    The value does not depend on the path before :math:`t`, which is
    why employee stock options granted at the money in the future and
    the legs of a cliquet are priced this way.

    Parameters
    ----------
    S : array_like
        Spot.
    alpha : array_like
        Strike as a fraction of the spot at the start date (1 for at the money).
    t_start, T : array_like
        Start date and expiry, ``t_start < T``.
    r, sigma : array_like
        Rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    Haug (2007), Section 4.3: S = 60, alpha = 1.1, t = 0.25, T = 1, r = 8%,
    b = 4%, sigma = 30%:

    >>> round(forward_start_price(60.0, 1.1, 0.25, 1.0, 0.08, 0.30, q=0.04), 3)
    4.406
    """
    S, t_start, T, q = (np.asarray(a, dtype=np.float64) for a in (S, t_start, T, q))
    nonnegative(S=S, t_start=t_start)
    return as_output(S * np.exp(-q * t_start) * np.asarray(black_scholes_price(1.0, alpha, T - t_start, r, sigma, q, option_type)))


def chooser_price(S: ArrayLike, K: ArrayLike, t_choose: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0) -> np.ndarray | float:
    """Rubinstein (1991) simple chooser: at :math:`t` the holder picks a call or a put, both struck at :math:`K` and expiring at :math:`T`.

    At :math:`t` the holder takes :math:`\\max(c, p) = c + (p - c)^+`, and by
    put-call parity :math:`p - c = K e^{-r(T-t)} - S_t e^{-q(T-t)}`, so

    .. math:: V = c(S, K, T) + e^{-q(T-t)}\\, p\\big(S, K e^{-(r-q)(T-t)}, t\\big).

    Parameters
    ----------
    S, K : array_like
        Spot and strike.
    t_choose, T : array_like
        Choice date and expiry, ``t_choose <= T``.
    r, sigma : array_like
        Rate and volatility.
    q : array_like
        Continuous dividend yield.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Haug (2007), Section 4.4.1: S = 50, K = 50, t = 0.25, T = 0.5, r = 8%,
    b = 8%, sigma = 25%:

    >>> round(chooser_price(50.0, 50.0, 0.25, 0.5, 0.08, 0.25), 4)
    6.1071
    """
    K, t, T, r, q = (np.asarray(a, dtype=np.float64) for a in (K, t_choose, T, r, q))
    tau = T - t
    call = np.asarray(black_scholes_price(S, K, T, r, sigma, q, "call"))
    put = np.asarray(black_scholes_price(S, K * np.exp(-(r - q) * tau), t, r, sigma, q, "put"))
    return as_output(call + np.exp(-q * tau) * put)
