"""Smile parametrisations and asymptotics: SABR, SVI and Lee's moment formula.

Hagan, Kumar, Lesniewski and Woodward (2002) gave the SABR model,

.. math::

   dF = \\alpha_t F^{\\beta}\\,dW_1, \\qquad d\\alpha = \\nu\\alpha\\,dW_2, \\qquad
   dW_1\\,dW_2 = \\rho\\,dt,

and an asymptotic formula for its Black implied volatility
(:func:`sabr_implied_vol`), which made it the market standard for
interest-rate smiles: :math:`\\beta` sets the backbone, :math:`\\rho` the
skew and :math:`\\nu` the curvature (:func:`sabr_calibrate`).

Gatheral (2004) proposed the stochastic volatility inspired (SVI)
parametrisation of the total implied variance
:math:`w(k) = \\sigma_{BS}^2 T` in log-moneyness :math:`k = \\ln(K/F)`,

.. math::

   w(k) = a + b\\left[\\rho (k - m) + \\sqrt{(k - m)^2 + \\sigma^2}\\right],

five numbers whose wings are linear, as they must be (:func:`svi_fit`).
Durrleman's function :func:`durrleman_g` is non-negative exactly when the
implied density is, which checks a fit for butterfly arbitrage.

Lee (2004) proved that the wings can be no steeper than linear and tied
the slope to the moments of the price: if :math:`\\tilde p` is the largest
:math:`p` with :math:`E[S_T^{1+p}] < \\infty`, then
:math:`\\limsup_{k\\to\\infty} w(k)/k = 2 - 4(\\sqrt{\\tilde p^2 + \\tilde p} - \\tilde p)`,
and the same holds on the left with negative moments (:func:`lee_wing_slope`).

References: P. S. Hagan, D. Kumar, A. S. Lesniewski and D. E. Woodward,
*Wilmott Magazine* (Sept. 2002) 84-108; J. Gatheral, "A parsimonious
arbitrage-free implied volatility parameterization," Global Derivatives
& Risk Management, Madrid (2004), and *The Volatility Surface* (Wiley,
2006), Chapter 3; R. W. Lee, *Mathematical Finance* 14 (2004) 469-480;
J. Gatheral and A. Jacquier, *Quantitative Finance* 14 (2014) 59-71.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from financialkit.options.utils.common import as_output
from financialkit.stochastic_volatility.core.base import SABRFit, SVIFit

__all__ = ["sabr_implied_vol", "sabr_calibrate", "svi_total_variance", "svi_fit", "durrleman_g", "lee_wing_slope"]


def sabr_implied_vol(F: ArrayLike, K: ArrayLike, T: ArrayLike, alpha: float, beta: float, rho: float, nu: float) -> np.ndarray | float:
    r"""Hagan et al.'s (2002) Black implied volatility of the SABR model.

    .. math::

       \sigma_B = \frac{\alpha}{(FK)^{(1-\beta)/2}\left[1 + \frac{(1-\beta)^2}{24}\ell^2 + \frac{(1-\beta)^4}{1920}\ell^4\right]}
       \frac{z}{x(z)}\left[1 + \left(\frac{(1-\beta)^2 \alpha^2}{24 (FK)^{1-\beta}}
       + \frac{\rho\beta\nu\alpha}{4 (FK)^{(1-\beta)/2}} + \frac{2 - 3\rho^2}{24}\nu^2\right) T\right],

    with :math:`\ell = \ln(F/K)`, :math:`z = (\nu/\alpha)(FK)^{(1-\beta)/2}\ell`
    and :math:`x(z) = \ln[(\sqrt{1 - 2\rho z + z^2} + z - \rho)/(1 - \rho)]`
    (:math:`z/x(z) = 1` at the money).

    Parameters
    ----------
    F, K, T : array_like
        Forward, strike and expiry.
    alpha, beta, rho, nu : float
        SABR parameters: initial volatility, CEV exponent in [0, 1],
        correlation and volatility of volatility.

    Returns
    -------
    float or ndarray

    Examples
    --------
    With :math:`\beta = 1` and no volatility of volatility it is
    lognormal with volatility :math:`\alpha`:

    >>> sabr_implied_vol(100.0, [80.0, 100.0, 120.0], 1.0, 0.2, 1.0, 0.0, 0.0)
    array([0.2, 0.2, 0.2])
    """
    F, K, T = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (F, K, T)))
    ob = 1.0 - beta
    fk = (F * K) ** (0.5 * ob)
    ell = np.log(F / K)
    z = nu / alpha * fk * ell
    small = np.abs(z) < 1e-8
    zs = np.where(small, 1.0, z)
    xz = np.log((np.sqrt(1.0 - 2.0 * rho * zs + zs * zs) + zs - rho) / (1.0 - rho))
    ratio = np.where(small, 1.0 - 0.5 * rho * z, zs / np.where(small, 1.0, xz))
    denom = fk * (1.0 + ob**2 / 24 * ell**2 + ob**4 / 1920 * ell**4)
    corr = 1.0 + (ob**2 * alpha**2 / (24 * fk**2) + rho * beta * nu * alpha / (4 * fk) + (2 - 3 * rho**2) * nu**2 / 24) * T
    return as_output(alpha / denom * ratio * corr)


def sabr_calibrate(F: float, strikes: ArrayLike, T: float, vols: ArrayLike, beta: float = 0.5) -> SABRFit:
    """Fit :math:`(\\alpha, \\rho, \\nu)` to a smile with :math:`\\beta` fixed, by :func:`scipy.optimize.least_squares`.

    Parameters
    ----------
    F : float
        Forward.
    strikes, vols : array_like, shape (n,)
        Strikes and market implied volatilities.
    T : float
        Expiry.
    beta : float
        CEV exponent, conventionally chosen rather than fitted.

    Returns
    -------
    SABRFit

    Examples
    --------
    >>> import numpy as np
    >>> K = np.linspace(70, 130, 13)
    >>> fit = sabr_calibrate(100.0, K, 1.0, sabr_implied_vol(100.0, K, 1.0, 2.0, 0.5, -0.3, 0.4))
    >>> round(fit.alpha, 4), round(fit.rho, 4), round(fit.nu, 4)
    (2.0, -0.3, 0.4)
    """
    K = np.asarray(strikes, dtype=np.float64)
    v = np.asarray(vols, dtype=np.float64)
    atm = float(np.interp(F, K, v)) if K.size > 1 else float(v[0])
    x0 = [atm * F ** (1.0 - beta), -0.2, 0.5]

    def resid(p: np.ndarray) -> np.ndarray:
        return np.asarray(sabr_implied_vol(F, K, T, p[0], beta, p[1], p[2])) - v

    sol = least_squares(resid, x0, bounds=([1e-8, -0.999, 1e-8], [np.inf, 0.999, np.inf]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    alpha, rho, nu = (float(a) for a in sol.x)
    return SABRFit(alpha, float(beta), rho, nu, float(F), float(T), float(np.sqrt(np.mean(sol.fun**2))))


def svi_total_variance(k: ArrayLike, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray | float:
    """Gatheral's raw SVI total implied variance :math:`w(k)`.

    Parameters
    ----------
    k : array_like
        Log-moneyness :math:`\\ln(K/F)`.
    a, b, rho, m, sigma : float
        Raw SVI parameters.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(svi_total_variance(0.0, 0.02, 0.1, -0.5, 0.0, 0.2), 12)
    0.04
    """
    k = np.asarray(k, dtype=np.float64)
    return as_output(a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2)))


def svi_fit(k: ArrayLike, w: ArrayLike, T: float) -> SVIFit:
    """Fit raw SVI to total implied variances by :func:`scipy.optimize.least_squares`.

    Several starting points are tried and the best fit kept. Bounds keep
    :math:`b \\ge 0`, :math:`|\\rho| < 1` and :math:`\\sigma > 0`.

    Parameters
    ----------
    k, w : array_like, shape (n,)
        Log-moneyness and total implied variance :math:`\\sigma_{BS}^2 T`.
    T : float
        Expiry, stored for :meth:`SVIFit.implied_vol`.

    Returns
    -------
    SVIFit

    Examples
    --------
    >>> import numpy as np
    >>> k = np.linspace(-0.5, 0.5, 21)
    >>> fit = svi_fit(k, svi_total_variance(k, 0.01, 0.12, -0.6, 0.05, 0.15), 1.0)
    >>> bool(fit.rmse < 1e-8)
    True
    """
    k = np.asarray(k, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    def resid(p: np.ndarray) -> np.ndarray:
        return np.asarray(svi_total_variance(k, *p)) - w

    lo = [-np.inf, 0.0, -0.999, 2 * k.min() - k.max(), 1e-4]
    hi = [np.inf, np.inf, 0.999, 2 * k.max() - k.min(), 10.0]
    best = None
    for rho0 in (-0.7, 0.0, 0.5):
        for sig0 in (0.05, 0.3):
            x0 = [float(w.min()) * 0.5, 0.1, rho0, float(k[np.argmin(w)]), sig0]
            sol = least_squares(resid, x0, bounds=(lo, hi), xtol=1e-15, ftol=1e-15, gtol=1e-15)
            if best is None or sol.cost < best.cost:
                best = sol
    assert best is not None
    a, b, rho, m, sigma = (float(v) for v in best.x)
    return SVIFit(a, b, rho, m, sigma, float(T), float(np.sqrt(np.mean(best.fun**2))))


def durrleman_g(k: ArrayLike, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray | float:
    r"""Durrleman's condition for SVI: the implied density is :math:`g(k)\,n(d_2)/\sqrt{w}`.

    .. math::

       g(k) = \left(1 - \frac{k w'}{2 w}\right)^2 - \frac{w'^2}{4}\left(\frac1w + \frac14\right) + \frac{w''}{2},

    so a slice is free of butterfly arbitrage exactly when :math:`g \ge 0`
    (Gatheral and Jacquier 2014).

    Parameters
    ----------
    k : array_like
    a, b, rho, m, sigma : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> bool(durrleman_g(0.0, 0.04, 0.1, -0.3, 0.0, 0.2) > 0)
    True
    """
    k = np.asarray(k, dtype=np.float64)
    s = np.sqrt((k - m) ** 2 + sigma**2)
    w = a + b * (rho * (k - m) + s)
    w1 = b * (rho + (k - m) / s)
    w2 = b * sigma**2 / s**3
    return as_output((1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2)


def lee_wing_slope(p: ArrayLike) -> np.ndarray | float:
    r"""Lee's (2004) moment formula: the limiting wing slope of :math:`w(k)/|k|`.

    .. math:: \beta = 2 - 4\left(\sqrt{p^2 + p} - p\right) \in [0, 2],

    with :math:`p` the critical moment: :math:`\tilde p` for the right
    wing (:math:`E[S_T^{1+p}] < \infty`), :math:`\tilde q` for the left
    (:math:`E[S_T^{-q}] < \infty`). Infinite moments of all orders give a
    slope of 0; no moments beyond the first give the maximum, 2.

    Parameters
    ----------
    p : array_like
        Critical moment, non-negative (``inf`` allowed).

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> lee_wing_slope([0.0, np.inf])
    array([2., 0.])
    """
    p = np.asarray(p, dtype=np.float64)
    fin = np.isfinite(p)
    pf = np.where(fin, p, 0.0)
    return as_output(np.where(fin, 2.0 - 4.0 * (np.sqrt(pf * pf + pf) - pf), 0.0))
