"""Options on two assets: exchange options, options on the maximum or minimum, and quantos.

* :func:`margrabe_price` -- Margrabe (1978): the option to exchange one
  asset for another is a Black-Scholes call with the second asset as
  numeraire, the volatility of the ratio and no interest rate.
* :func:`stulz_price` -- Stulz (1982): calls and puts on the maximum or
  the minimum of two lognormal assets, with the bivariate normal.
* :func:`quanto_price` -- Reiner (1992): a foreign-asset option paid in
  domestic currency at a fixed exchange rate; the correlation between
  asset and currency shifts the asset's drift by :math:`-\\rho\\sigma_S\\sigma_X`.

References: W. Margrabe, "The Value of an Option to Exchange One Asset
for Another," *J. Finance* 33 (1978) 177-186; R. M. Stulz, "Options on
the Minimum or the Maximum of Two Risky Assets," *J. Financial
Economics* 10 (1982) 161-185; E. Reiner, "Quanto Mechanics," *Risk* 5(3)
(1992) 59-63; E. G. Haug, *The Complete Guide to Option Pricing
Formulas*, 2nd ed. (McGraw-Hill, 2007), Sections 4.8.1, 4.8.4 and 5.16.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit._validation import nonnegative, one_of, positive
from financialkit._validation import option_type as check_option_type
from financialkit.exotic_options.utils.numerics import bivariate_normal_cdf
from financialkit.options import black76_price, black_scholes_price
from financialkit.options.utils.common import as_output

__all__ = ["margrabe_price", "stulz_price", "quanto_price"]


def _spread_vol(sigma1, sigma2, rho):
    return np.sqrt(sigma1**2 + sigma2**2 - 2.0 * rho * sigma1 * sigma2)


def margrabe_price(
    S1: ArrayLike, S2: ArrayLike, T: ArrayLike, sigma1: ArrayLike, sigma2: ArrayLike, rho: ArrayLike, q1: ArrayLike = 0.0, q2: ArrayLike = 0.0
) -> np.ndarray | float:
    """Margrabe (1978) value of the right to receive asset 1 in exchange for asset 2 at :math:`T`: :math:`(S_1 - S_2)^+`.

    Measured in units of asset 2 the payoff is a call on :math:`S_1/S_2`
    struck at 1, and the ratio has volatility
    :math:`\\sigma = \\sqrt{\\sigma_1^2 + \\sigma_2^2 - 2\\rho\\sigma_1\\sigma_2}`:

    .. math:: V = S_1 e^{-q_1 T} N(d_1) - S_2 e^{-q_2 T} N(d_2), \\qquad
              d_{1,2} = \\frac{\\ln(S_1 e^{-q_1 T} / S_2 e^{-q_2 T}) \\pm \\sigma^2 T/2}{\\sigma\\sqrt T}.

    The interest rate drops out. This is :func:`financialkit.options.black_scholes_price`
    with spot :math:`S_1 e^{-q_1 T}`, strike :math:`S_2 e^{-q_2 T}` and zero rate.

    Parameters
    ----------
    S1, S2 : array_like
        Prices of the asset received and the asset given up.
    T : array_like
        Maturity.
    sigma1, sigma2 : array_like
        Volatilities.
    rho : array_like
        Correlation of the two assets' returns.
    q1, q2 : array_like
        Continuous dividend yields.

    Returns
    -------
    float or ndarray

    Examples
    --------
    When asset 2 is riskless cash growing at the rate :math:`r` (zero
    volatility, "dividend yield" :math:`r`) the exchange option is the
    Black-Scholes call:

    >>> from financialkit.options import black_scholes_price
    >>> m = margrabe_price(42.0, 40.0, 0.5, 0.2, 0.0, 0.0, q2=0.10)
    >>> bool(abs(m - black_scholes_price(42.0, 40.0, 0.5, 0.10, 0.2)) < 1e-12)
    True
    """
    S1, S2, T, q1, q2 = (np.asarray(a, dtype=np.float64) for a in (S1, S2, T, q1, q2))
    nonnegative(S1=S1, S2=S2)
    return black_scholes_price(S1 * np.exp(-q1 * T), S2 * np.exp(-q2 * T), T, 0.0, _spread_vol(sigma1, sigma2, rho))


def stulz_price(
    S1: ArrayLike,
    S2: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma1: ArrayLike,
    sigma2: ArrayLike,
    rho: ArrayLike,
    q1: ArrayLike = 0.0,
    q2: ArrayLike = 0.0,
    option_type: str = "call",
    kind: str = "max",
) -> np.ndarray | float:
    """Stulz (1982) option on the maximum or minimum of two assets: :math:`(\\max(S_1, S_2) - K)^+` and its relatives.

    With :math:`\\sigma` the ratio volatility,
    :math:`\\rho_1 = (\\sigma_1 - \\rho\\sigma_2)/\\sigma`,
    :math:`\\rho_2 = (\\sigma_2 - \\rho\\sigma_1)/\\sigma`,
    :math:`d = [\\ln(S_1/S_2) + (b_1 - b_2 + \\sigma^2/2)T]/(\\sigma\\sqrt T)` and
    :math:`y_i = [\\ln(S_i/K) + (b_i + \\sigma_i^2/2)T]/(\\sigma_i\\sqrt T)`,
    the call on the minimum is

    .. math:: S_1 e^{-q_1 T} M(y_1, -d; -\\rho_1) + S_2 e^{-q_2 T} M(y_2, d - \\sigma\\sqrt T; -\\rho_2)
              - K e^{-rT} M(y_1 - \\sigma_1\\sqrt T, y_2 - \\sigma_2\\sqrt T; \\rho),

    the call on the maximum is its counterpart (Haug 2007, Sec. 4.8.4),
    and the puts follow from parity with the zero-strike call, which is a
    Margrabe exchange option. Since
    :math:`\\max + \\min = S_1 + S_2`, the calls on the maximum and the
    minimum add up to the two vanilla calls. At ``T = 0``, or with both
    volatilities zero, the price is the discounted payoff on the forwards;
    with only one of them zero, or a zero ratio volatility, it raises
    :class:`ValueError`.

    Parameters
    ----------
    S1, S2, K, T, r : array_like
        Spots, strike, maturity and rate.
    sigma1, sigma2 : array_like
        Volatilities.
    rho : array_like
        Correlation, :math:`|\\rho| < 1`.
    q1, q2 : array_like
        Continuous dividend yields.
    option_type : {"call", "put"}
    kind : {"max", "min"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> from financialkit.options import black_scholes_price
    >>> args = (100.0, 105.0, 98.0, 0.5, 0.05, 0.11, 0.16, 0.63)
    >>> total = stulz_price(*args, kind="max") + stulz_price(*args, kind="min")
    >>> vanilla = black_scholes_price(100.0, 98.0, 0.5, 0.05, 0.11) + black_scholes_price(105.0, 98.0, 0.5, 0.05, 0.16)
    >>> bool(abs(total - vanilla) < 1e-10)
    True
    """
    phi = check_option_type(option_type)
    one_of("kind", kind, ("max", "min"))
    S1, S2, K, T, r, s1, s2, rho, q1, q2 = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (S1, S2, K, T, r, sigma1, sigma2, rho, q1, q2)))
    nonnegative(K=K, T=T, sigma1=s1, sigma2=s2)
    positive(S1=S1, S2=S2)
    s = _spread_vol(s1, s2, rho)
    riskless = (T == 0) | ((s1 == 0) & (s2 == 0))
    if np.any(~riskless & ((s1 == 0) | (s2 == 0) | (s == 0))):
        raise ValueError("stulz_price needs positive sigma1, sigma2 and ratio volatility, unless both volatilities are zero or T = 0")
    F1, F2 = S1 * np.exp((r - q1) * T), S2 * np.exp((r - q2) * T)
    extreme = np.maximum(F1, F2) if kind == "max" else np.minimum(F1, F2)
    deterministic = np.exp(-r * T) * np.maximum(phi * (extreme - K), 0.0)
    T, s1, s2, s = (np.where(riskless, 1.0, x) for x in (T, s1, s2, s))
    rho1, rho2 = (s1 - rho * s2) / s, (s2 - rho * s1) / s
    sqT = np.sqrt(T)
    d = (np.log(S1 / S2) + (q2 - q1 + 0.5 * s**2) * T) / (s * sqT)
    with np.errstate(divide="ignore"):  # K = 0 sends y1 and y2 to +inf, which bivariate_normal_cdf handles
        y1 = (np.log(S1 / K) + (r - q1 + 0.5 * s1**2) * T) / (s1 * sqT)
        y2 = (np.log(S2 / K) + (r - q2 + 0.5 * s2**2) * T) / (s2 * sqT)
    A1, A2, disc = S1 * np.exp(-q1 * T), S2 * np.exp(-q2 * T), K * np.exp(-r * T)
    M = bivariate_normal_cdf
    if kind == "min":
        call = A1 * M(y1, -d, -rho1) + A2 * M(y2, d - s * sqT, -rho2) - disc * M(y1 - s1 * sqT, y2 - s2 * sqT, rho)
    else:
        call = A1 * M(y1, d, rho1) + A2 * M(y2, -d + s * sqT, rho2) - disc * (1.0 - M(-y1 + s1 * sqT, -y2 + s2 * sqT, rho))
    if phi == 1:
        return as_output(np.where(riskless, deterministic, call))
    exchange = np.asarray(margrabe_price(S1, S2, T, s1, s2, rho, q1, q2))
    zero_strike = A1 - exchange if kind == "min" else A2 + exchange
    return as_output(np.where(riskless, deterministic, disc - zero_strike + call))


def quanto_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r_d: ArrayLike,
    r_f: ArrayLike,
    sigma_S: ArrayLike,
    sigma_X: ArrayLike,
    rho: ArrayLike,
    q: ArrayLike = 0.0,
    fixed_rate: ArrayLike = 1.0,
    option_type: str = "call",
) -> np.ndarray | float:
    """Reiner (1992) quanto option: a foreign-currency payoff :math:`(S_T - K)^+` paid in domestic currency at a fixed rate :math:`\\bar X`.

    Under the domestic risk-neutral measure the foreign asset drifts at
    :math:`r_f - q - \\rho\\sigma_S\\sigma_X`, where :math:`\\rho` is the
    correlation between the asset and the exchange rate (domestic units
    per foreign unit). The quanto is Black's formula on the adjusted forward,
    discounted domestically:

    .. math:: V = \\bar X e^{-r_d T}\\left[\\hat F N(d_1) - K N(d_2)\\right], \\qquad
              \\hat F = S e^{(r_f - q - \\rho\\sigma_S\\sigma_X)T}.

    Parameters
    ----------
    S, K : array_like
        Foreign asset price and strike, in foreign currency.
    T : array_like
        Maturity.
    r_d, r_f : array_like
        Domestic and foreign rates.
    sigma_S, sigma_X : array_like
        Volatilities of the asset and of the exchange rate.
    rho : array_like
        Correlation between the asset and the exchange rate.
    q : array_like
        Continuous dividend yield of the asset.
    fixed_rate : array_like
        Fixed conversion rate :math:`\\bar X`, domestic per foreign.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    With no correlation the quanto is the foreign option at the fixed
    rate, re-discounted at the domestic rate:

    >>> import numpy as np
    >>> from financialkit.options import black_scholes_price
    >>> v = quanto_price(100.0, 105.0, 0.5, 0.08, 0.05, 0.2, 0.1, 0.0, q=0.04, fixed_rate=1.5)
    >>> bool(abs(v - 1.5 * np.exp(-0.03 * 0.5) * black_scholes_price(100.0, 105.0, 0.5, 0.05, 0.2, q=0.04)) < 1e-12)
    True
    """
    S, T, r_f, sigma_S, sigma_X, rho, q = (np.asarray(a, dtype=np.float64) for a in (S, T, r_f, sigma_S, sigma_X, rho, q))
    nonnegative(S=S)
    F = S * np.exp((r_f - q - rho * sigma_S * sigma_X) * T)
    return as_output(np.asarray(fixed_rate, dtype=np.float64) * np.asarray(black76_price(F, K, T, r_d, sigma_S, option_type)))
