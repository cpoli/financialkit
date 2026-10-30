"""The vanna-volga method for the FX smile (Castagna and Mercurio, 2007).

FX option markets quote three volatilities per expiry: at the money,
the 25-delta risk reversal and the 25-delta butterfly, which together
give three pillar strikes :math:`K_1 < K_2 < K_3`. Traders filled in the
rest of the smile by hedging: a European option at any strike :math:`K`
is priced at Black-Scholes with the at-the-money volatility, plus the cost
of the three pillar options whose weights :math:`x_i(K)` replicate its
vega, vanna and volga,

.. math:: C(K) = C_{BS}(K, \\sigma_2) + \\sum_{i=1}^{3} x_i(K)
          \\left[C_{\\text{mkt}}(K_i) - C_{BS}(K_i, \\sigma_2)\\right],
          \\qquad x_i(K) = \\frac{\\mathcal V(K)}{\\mathcal V(K_i)}
          \\prod_{j \\ne i} \\frac{\\ln(K_j / K)}{\\ln(K_j / K_i)}.

Castagna and Mercurio derived these weights in closed form, showed that
the price reproduces the three quotes exactly, and gave first- and
second-order approximations of its implied volatility:

.. math:: \\sigma(K) \\approx \\sum_i y_i(K)\\sigma_i, \\qquad
          y_i(K) = \\prod_{j \\ne i} \\frac{\\ln(K_j / K)}{\\ln(K_j / K_i)},

the first order being a quadratic in :math:`\\ln K` through the pillars.

References: A. Castagna and F. Mercurio, "The Vanna-Volga Method for
Implied Volatilities," *Risk* 20(1) (January 2007) 106-111; U. Wystup,
*FX Options and Structured Products*, Wiley (2006), Section 1.5;
Hull, *Options, Futures, and Other Derivatives*, 10th ed., Section 17.11.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from financialkit._validation import one_of, positive
from financialkit._validation import option_type as check_option_type
from financialkit.options.core.base import SmileResult
from financialkit.options.systems.closed_form import garman_kohlhagen_price
from financialkit.options.systems.implied_vol import implied_volatility
from financialkit.options.utils.common import as_output

__all__ = ["fx_pillar_strikes", "vanna_volga_price", "vanna_volga_smile"]

_METHODS = ("price", "first_order", "second_order")


def fx_pillar_strikes(S: float, T: float, r_d: float, r_f: float, vol_25p: float, vol_atm: float, vol_25c: float) -> np.ndarray:
    r"""Strikes of the 25-delta put, the delta-neutral straddle and the 25-delta call.

    With spot deltas :math:`\pm e^{-r_f T} N(\pm d_1) = \pm 1/4` (Castagna
    and Mercurio's convention),

    .. math:: K_{1,3} = S e^{\mp\alpha\sigma_{1,3}\sqrt{T} + (r_d - r_f + \sigma_{1,3}^2/2)T},
              \quad K_2 = S e^{(r_d - r_f + \sigma_2^2/2)T},
              \quad \alpha = -N^{-1}\!\left(\tfrac14 e^{r_f T}\right).

    Parameters
    ----------
    S, T, r_d, r_f : float
        Spot, maturity and the domestic and foreign rates.
    vol_25p, vol_atm, vol_25c : float
        Volatilities of the 25-delta put, the at-the-money straddle and the
        25-delta call.

    Returns
    -------
    ndarray, shape (3,)

    Examples
    --------
    >>> K = fx_pillar_strikes(1.20, 0.25, 0.03, 0.01, 0.105, 0.10, 0.098)
    >>> bool(K[0] < 1.20 * 1.005 < K[2])
    True
    """
    positive(S=S, T=T, vol_25p=vol_25p, vol_atm=vol_atm, vol_25c=vol_25c)
    alpha = -norm.ppf(0.25 * np.exp(r_f * T))
    vols = np.array([vol_25p, vol_atm, vol_25c])
    shift = np.array([-alpha, 0.0, alpha]) * vols * np.sqrt(T)
    return S * np.exp(shift + (r_d - r_f + 0.5 * vols**2) * T)


def _log_weights(K: np.ndarray, pillars: np.ndarray) -> list[np.ndarray]:
    """Lagrange weights :math:`y_i(K)` in :math:`\\ln K` through the three pillars."""
    lk, lp = np.log(K), np.log(pillars)
    return [np.prod([(lp[j] - lk) / (lp[j] - lp[i]) for j in range(3) if j != i], axis=0) for i in range(3)]


def _vega(S, K, T, r_d, r_f, sigma):
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-r_f * T) * np.sqrt(T) * norm.pdf(d1), d1


def vanna_volga_price(
    K: ArrayLike, S: float, T: float, r_d: float, r_f: float, pillar_strikes: ArrayLike, pillar_vols: ArrayLike, option_type: str = "call"
) -> np.ndarray | float:
    r"""Castagna-Mercurio (2007) vanna-volga price of a European FX option.

    The Garman-Kohlhagen price at the middle pillar volatility, corrected
    by the market cost of the three-option hedge that matches the option's
    vega, vanna and volga (see the module docstring). It reproduces the
    three pillar quotes exactly.

    Parameters
    ----------
    K : array_like
        Strikes.
    S, T, r_d, r_f : float
        Spot, maturity and the domestic and foreign rates.
    pillar_strikes : array_like, shape (3,)
        Increasing pillar strikes :math:`K_1 < K_2 < K_3`, usually from
        :func:`fx_pillar_strikes`.
    pillar_vols : array_like, shape (3,)
        Market volatilities at the pillars.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    A flat smile leaves Garman-Kohlhagen unchanged:

    >>> import numpy as np
    >>> pillars = fx_pillar_strikes(1.2, 0.5, 0.03, 0.01, 0.1, 0.1, 0.1)
    >>> c = vanna_volga_price(1.25, 1.2, 0.5, 0.03, 0.01, pillars, [0.1, 0.1, 0.1])
    >>> bool(np.isclose(c, garman_kohlhagen_price(1.2, 1.25, 0.5, 0.03, 0.01, 0.1)))
    True
    """
    check_option_type(option_type)
    K = np.asarray(K, dtype=np.float64)
    pillars, vols = _pillars(pillar_strikes, pillar_vols)
    positive(K=K, S=S, T=T)
    sigma2 = vols[1]
    vega_K, _ = _vega(S, K, T, r_d, r_f, sigma2)
    price = np.asarray(garman_kohlhagen_price(S, K, T, r_d, r_f, sigma2, option_type), dtype=np.float64)
    for i, y in enumerate(_log_weights(K, pillars)):
        vega_i, _ = _vega(S, pillars[i], T, r_d, r_f, sigma2)
        market = garman_kohlhagen_price(S, pillars[i], T, r_d, r_f, vols[i], option_type)
        flat = garman_kohlhagen_price(S, pillars[i], T, r_d, r_f, sigma2, option_type)
        price = price + vega_K / vega_i * y * (market - flat)
    return as_output(price)


def _pillars(pillar_strikes: ArrayLike, pillar_vols: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    pillars, vols = np.asarray(pillar_strikes, dtype=np.float64), np.asarray(pillar_vols, dtype=np.float64)
    if pillars.shape != (3,) or vols.shape != (3,):
        raise ValueError(f"need three pillar strikes and three pillar vols, got shapes {pillars.shape} and {vols.shape}")
    positive(pillar_strikes=pillars, pillar_vols=vols)
    if not np.all(np.diff(pillars) > 0):
        raise ValueError(f"pillar strikes must be increasing, got {pillars}")
    return pillars, vols


def vanna_volga_smile(
    strikes: ArrayLike, S: float, T: float, r_d: float, r_f: float, pillar_strikes: ArrayLike, pillar_vols: ArrayLike, method: str = "price"
) -> SmileResult:
    r"""The vanna-volga implied-volatility smile.

    Parameters
    ----------
    strikes : array_like, shape (n,)
        Strikes at which to read the smile.
    S, T, r_d, r_f : float
        Spot, maturity and the domestic and foreign rates.
    pillar_strikes, pillar_vols : array_like, shape (3,)
        Increasing pillar strikes and their market volatilities.
    method : {"price", "first_order", "second_order"}
        ``"price"`` inverts :func:`vanna_volga_price` (out-of-the-money
        options: puts below the forward, calls above); the others are
        Castagna and Mercurio's approximations, the first a quadratic in
        :math:`\ln K` and the second

        .. math:: \sigma(K) \approx \sigma_2 + \frac{-\sigma_2 + \sqrt{\sigma_2^2
                  + d_1 d_2 (2\sigma_2 D_1 + D_2)}}{d_1 d_2},

        with :math:`D_1 = \sum_i y_i\sigma_i - \sigma_2` and
        :math:`D_2 = \sum_i y_i d_1(K_i) d_2(K_i)(\sigma_i - \sigma_2)^2`, all
        :math:`d`'s at :math:`\sigma_2`. Where the square root's argument is
        negative, far in the wings, it returns ``nan``.

    Returns
    -------
    SmileResult

    Examples
    --------
    All three methods return the quotes at the pillars:

    >>> import numpy as np
    >>> vols = [0.105, 0.10, 0.098]
    >>> pillars = fx_pillar_strikes(1.2, 0.5, 0.03, 0.01, *vols)
    >>> [bool(np.allclose(vanna_volga_smile(pillars, 1.2, 0.5, 0.03, 0.01, pillars, vols, m).vols, vols))
    ...  for m in ("price", "first_order", "second_order")]
    [True, True, True]
    """
    one_of("method", method, _METHODS)
    strikes = np.asarray(strikes, dtype=np.float64)
    pillars, vols = _pillars(pillar_strikes, pillar_vols)
    positive(strikes=strikes, S=S, T=T)
    forward = float(S * np.exp((r_d - r_f) * T))
    y = _log_weights(strikes, pillars)
    first = sum(yi * vi for yi, vi in zip(y, vols, strict=True))
    if method == "price":
        out = np.empty(strikes.shape)
        for option_type, mask in (("put", strikes < forward), ("call", strikes >= forward)):
            if np.any(mask):
                price = vanna_volga_price(strikes[mask], S, T, r_d, r_f, pillars, vols, option_type)
                out[mask] = implied_volatility(price, S, strikes[mask], T, r_d, r_f, option_type, method="jaeckel")
    elif method == "first_order":
        out = first
    else:
        sigma2, vt = vols[1], vols[1] * np.sqrt(T)

        def dd(k):
            d1 = (np.log(S / k) + (r_d - r_f + 0.5 * sigma2**2) * T) / vt
            return d1 * (d1 - vt)

        D1 = first - sigma2
        D2 = sum(yi * dd(pillars[i]) * (vols[i] - sigma2) ** 2 for i, yi in enumerate(y))
        dk = dd(strikes)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = sigma2 + (-sigma2 + np.sqrt(sigma2**2 + dk * (2 * sigma2 * D1 + D2))) / dk
        out = np.where(np.abs(dk) < 1e-12, sigma2 + D1, out)  # limit at d1 d2 = 0
    return SmileResult(strikes=strikes, vols=np.asarray(out, dtype=np.float64), forward=forward, T=float(T))
