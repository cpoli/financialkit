"""Duffie and Singleton's recovery of market value, and the other recovery conventions.

Duffie and Singleton (1999) assumed that at default a claim loses a
fraction :math:`L` of its own pre-default market value. Then a
defaultable claim is priced like a riskless one with the short rate
replaced by the default-adjusted rate :math:`r + \\lambda L`:

.. math:: P^d(0, T) = E\\left[e^{-\\int_0^T (r_s + \\lambda_s L)\\,ds}\\right].

With a constant intensity this is :math:`e^{-(r + \\lambda L) T}`; with a
Cox-Ingersoll-Ross intensity the scaled intensity :math:`\\lambda L` is
again CIR, so the bond has the CIR closed form. The two older
conventions, recovery of treasury (Jarrow and Turnbull 1995) and
recovery of face value, are given for comparison. The riskless rate
``r`` is a number or a ``fixed_income`` yield curve; with deterministic
rates the riskless discount factor :math:`P(0, T)` simply multiplies
the default-adjusted one.

References: D. Duffie and K. J. Singleton, "Modeling Term Structures of
Defaultable Bonds," *Rev. Financial Stud.* 12 (1999) 687-720; D. Lando,
*Credit Risk Modeling* (Princeton, 2004), Ch. 5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate

from financialkit.credit.utils.discounting import discount_factors
from financialkit.fixed_income import CurveLike, YieldCurve
from financialkit.interest_rate_models import CIRModel
from financialkit.options.utils.common import as_output

__all__ = ["risky_zero_coupon", "duffie_singleton_cir_zero"]


def risky_zero_coupon(T: ArrayLike, r: ArrayLike | YieldCurve, hazard: ArrayLike, recovery: ArrayLike = 0.4, convention: str = "rmv") -> np.ndarray | float:
    """Defaultable zero-coupon bond with a constant hazard rate, under three recovery conventions.

    - ``"rmv"`` (recovery of market value): :math:`e^{-(r + \\lambda (1 - R)) T}`;
    - ``"rt"`` (recovery of treasury): :math:`e^{-rT}[R + (1 - R) e^{-\\lambda T}]`;
    - ``"rfv"`` (recovery of face value, paid at default):
      :math:`e^{-(r + \\lambda) T} + R \\frac{\\lambda}{r + \\lambda}(1 - e^{-(r + \\lambda) T})`.

    With a yield curve, :math:`e^{-rT}` becomes :math:`P(0, T)` and the
    recovery-of-face-value annuity :math:`\\int_0^T P(0, u) e^{-\\lambda u} du`
    is integrated with :func:`scipy.integrate.quad_vec`.

    Parameters
    ----------
    T, hazard, recovery : array_like
        Maturity, constant intensity and recovery fraction.
    r : array_like or YieldCurve
        Continuously compounded risk-free rate, or a ``fixed_income`` discount curve.
    convention : {"rmv", "rt", "rfv"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    With no recovery the three conventions agree:

    >>> [round(risky_zero_coupon(5.0, 0.05, 0.02, 0.0, c), 6) for c in ("rmv", "rt", "rfv")]
    [0.704688, 0.704688, 0.704688]
    """
    T, h, R = (np.asarray(a, dtype=np.float64) for a in (T, hazard, recovery))
    Z = discount_factors(r, T)
    if convention == "rmv":
        out = Z * np.exp(-h * (1.0 - R) * T)
    elif convention == "rt":
        out = Z * (R + (1.0 - R) * np.exp(-h * T))
    elif convention == "rfv":
        if isinstance(r, YieldCurve):
            # int_0^T P(0, u) e^{-h u} du, substituting u = sT so one quad_vec covers every T.
            annuity = integrate.quad_vec(lambda s: T * r.discount(s * T) * np.exp(-h * s * T), 0.0, 1.0, epsabs=1e-13, epsrel=1e-12)[0]
        else:
            k = np.asarray(r, dtype=np.float64) + h
            with np.errstate(divide="ignore", invalid="ignore"):
                annuity = np.where(k > 0, -np.expm1(-k * T) / np.where(k > 0, k, 1.0), T)
        out = Z * np.exp(-h * T) + R * h * annuity
    else:
        raise ValueError(f'convention must be "rmv", "rt" or "rfv", got {convention!r}')
    return as_output(out)


def duffie_singleton_cir_zero(
    T: ArrayLike, r: CurveLike, lambda0: float, kappa: float, theta: float, sigma: float, recovery: float = 0.4
) -> np.ndarray | float:
    """Recovery-of-market-value bond with a CIR default intensity and deterministic riskless rates.

    The loss-adjusted intensity :math:`L\\lambda` (with :math:`L = 1 - R`)
    follows a CIR process with long-run mean :math:`L\\theta` and volatility
    :math:`\\sigma\\sqrt{L}`, so

    .. math:: P^d(0, T) = P(0, T) A(T) e^{-B(T) L \\lambda_0},

    with :math:`A, B` from
    :class:`financialkit.interest_rate_models.CIRModel`.

    Parameters
    ----------
    T : array_like
        Maturities.
    r : float or YieldCurve
        Continuously compounded risk-free rate, or a ``fixed_income`` discount curve.
    lambda0, kappa, theta, sigma : float
        Initial intensity, mean-reversion speed, long-run mean and volatility of the intensity.
    recovery : float
        Recovery as a fraction of market value.

    Returns
    -------
    float or ndarray

    Examples
    --------
    With the intensity starting at its mean and almost no volatility, the bond is the constant-hazard one:

    >>> a = duffie_singleton_cir_zero(5.0, 0.05, 0.02, 1.0, 0.02, 1e-3, 0.4)
    >>> b = risky_zero_coupon(5.0, 0.05, 0.02, 0.4, "rmv")
    >>> abs(a - b) < 1e-6
    True
    """
    L = 1.0 - recovery
    T = np.asarray(T, dtype=np.float64)
    model = CIRModel(L * lambda0, kappa, L * theta, sigma * np.sqrt(L))
    return as_output(discount_factors(r, T) * np.asarray(model.zero_coupon_bond(T), dtype=np.float64))
