"""Credit default swaps: O'Kane and Turnbull's legs, the par spread, and the ISDA hazard bootstrap.

A CDS protection buyer pays a spread :math:`s` on the notional until
maturity or default, and receives :math:`1 - R` at default. With a
survival curve :math:`Q` and discount factors :math:`Z`, O'Kane and
Turnbull (2003) write the two legs as

.. math::

   \\text{Protection} = (1 - R) \\int_0^T Z(u)\\,(-dQ(u)), \\qquad
   \\text{RPV01} = \\sum_i \\Delta_i Z(t_i) Q(t_i)
   + \\int_0^T (u - t_{i(u)-1}) Z(u)\\,(-dQ(u)),

the second integral being the premium accrued at default, and
approximate both integrals on a grid with defaults at mid-step. The par
spread is :math:`\\text{Protection} / \\text{RPV01}`; for a flat hazard
and continuous premium it is exactly :math:`\\lambda (1 - R)`.

Since the 2009 "Big Bang", contracts trade with a fixed coupon (100 or
500 bp) plus an upfront payment, and the ISDA standard model bootstraps
a piecewise-constant hazard curve from a term structure of quotes,
solving one maturity at a time. Here the riskless curve is flat and
payments fall on an even grid; the ISDA model's day counts and IMM
dates are not modelled.

References: D. O'Kane and S. Turnbull, *Valuation of Credit Default
Swaps*, Lehman Brothers Fixed Income Quantitative Credit Research
(2003); ISDA, *CDS Standard Model* (2009); J. C. Hull, *Options,
Futures, and Other Derivatives*, 10th ed. (2018), Section 25.2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from financialkit.credit.core.base import CDSResult, SurvivalCurve
from financialkit.credit.systems.hazard import HazardCurve

__all__ = ["cds_legs", "cds_par_spread", "cds_upfront", "upfront_to_spread", "bootstrap_hazard_curve"]


def _payment_times(maturity: float, frequency: int) -> np.ndarray:
    n = max(int(np.ceil(maturity * frequency - 1e-9)), 1)
    return maturity - np.arange(n - 1, -1, -1) / frequency


def cds_legs(curve: SurvivalCurve, maturity: float, r: float, recovery: float = 0.4, frequency: int = 4, steps_per_year: int = 52) -> CDSResult:
    """Value both legs of a CDS per unit notional, starting today.

    Parameters
    ----------
    curve : SurvivalCurve
        Risk-neutral survival curve of the reference entity.
    maturity : float
        Years to maturity.
    r : float
        Flat continuously compounded risk-free rate.
    recovery : float
        Recovery rate :math:`R`.
    frequency : int
        Premium payments per year (4 for the standard quarterly schedule).
    steps_per_year : int
        Grid for the default integrals; each premium period gets at least one step.

    Returns
    -------
    CDSResult

    Examples
    --------
    Hull's (10th ed., Section 25.2) setting: hazard 2%, recovery 40%,
    rate 5%, annual payments, defaults half-way through each year.

    >>> res = cds_legs(HazardCurve.flat(0.02), 5.0, 0.05, 0.4, frequency=1, steps_per_year=1)
    >>> round(res.rpv01 - res.accrued_on_default, 4), round(res.protection_leg, 4)
    (4.0728, 0.0506)
    >>> round(1e4 * res.par_spread, 1)
    123.0
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    pay = _payment_times(maturity, frequency)
    starts = np.r_[0.0, pay[:-1]]
    accrual = pay - starts
    premium = float(np.sum(accrual * np.exp(-r * pay) * np.asarray(curve.survival(pay))))
    accrued = 0.0
    protection = 0.0
    for a, b in zip(starts, pay, strict=True):
        m = max(int(np.ceil((b - a) * steps_per_year - 1e-9)), 1)
        u = np.linspace(a, b, m + 1)
        dq = -np.diff(np.asarray(curve.survival(u), dtype=np.float64))
        mid = 0.5 * (u[:-1] + u[1:])
        z = np.exp(-r * mid)
        accrued += float(np.sum((mid - a) * z * dq))
        protection += float(np.sum(z * dq))
    protection *= 1.0 - recovery
    rpv01 = premium + accrued
    return CDSResult(par_spread=protection / rpv01, protection_leg=protection, rpv01=rpv01, accrued_on_default=accrued, payment_times=pay)


def cds_par_spread(curve: SurvivalCurve, maturity: float, r: float, recovery: float = 0.4, frequency: int = 4, steps_per_year: int = 52) -> float:
    """Par spread of a CDS, ``protection_leg / rpv01``.

    Parameters
    ----------
    curve, maturity, r, recovery, frequency, steps_per_year
        As in :func:`cds_legs`.

    Returns
    -------
    float

    Examples
    --------
    The credit triangle :math:`s = \\lambda(1 - R)` is the continuous-premium limit:

    >>> round(1e4 * cds_par_spread(HazardCurve.flat(0.02), 5.0, 0.05, 0.4, frequency=365, steps_per_year=365), 1)
    120.0
    """
    return cds_legs(curve, maturity, r, recovery, frequency, steps_per_year).par_spread


def cds_upfront(curve: SurvivalCurve, maturity: float, coupon: float, r: float, recovery: float = 0.4, frequency: int = 4, steps_per_year: int = 52) -> float:
    """Upfront paid by the protection buyer of a fixed-coupon CDS, per unit notional.

    ``protection_leg - coupon * rpv01``; negative when the buyer is paid.

    Parameters
    ----------
    curve, maturity, r, recovery, frequency, steps_per_year
        As in :func:`cds_legs`.
    coupon : float
        Standard running coupon, e.g. ``0.01`` or ``0.05``.

    Returns
    -------
    float
    """
    return cds_legs(curve, maturity, r, recovery, frequency, steps_per_year).mtm(coupon)


def upfront_to_spread(upfront: float, coupon: float, maturity: float, r: float, recovery: float = 0.4, frequency: int = 4, steps_per_year: int = 52) -> float:
    """ISDA quote conversion: the flat-hazard par spread equivalent to an upfront.

    Finds the flat hazard rate that reproduces ``upfront`` at ``coupon``
    and returns that curve's par spread.

    Parameters
    ----------
    upfront : float
        Upfront per unit notional.
    coupon, maturity, r, recovery, frequency, steps_per_year
        As in :func:`cds_upfront`.

    Returns
    -------
    float
    """

    def f(h):
        return cds_upfront(HazardCurve.flat(h), maturity, coupon, r, recovery, frequency, steps_per_year) - upfront

    h = optimize.brentq(f, 0.0, 10.0, xtol=1e-14)
    return cds_par_spread(HazardCurve.flat(h), maturity, r, recovery, frequency, steps_per_year)


def bootstrap_hazard_curve(
    maturities: ArrayLike, spreads: ArrayLike, r: float, recovery: float = 0.4, frequency: int = 4, steps_per_year: int = 52
) -> HazardCurve:
    """Bootstrap a piecewise-constant hazard curve from CDS par spreads.

    Solves the maturities in increasing order with
    :func:`scipy.optimize.brentq`, each new hazard segment chosen so the
    CDS to that maturity reprices at its quote while the earlier segments
    stay fixed, as in the ISDA standard model.

    Parameters
    ----------
    maturities : array_like, shape (n,)
        Increasing CDS maturities, e.g. ``[1, 3, 5, 7, 10]``.
    spreads : array_like, shape (n,)
        Par spreads.
    r, recovery, frequency, steps_per_year
        As in :func:`cds_legs`.

    Returns
    -------
    HazardCurve

    Examples
    --------
    >>> c = bootstrap_hazard_curve([1, 3, 5], [0.006, 0.009, 0.012], 0.03)
    >>> [round(1e4 * cds_par_spread(c, T, 0.03), 6) for T in (1, 3, 5)]
    [60.0, 90.0, 120.0]
    """
    T = np.asarray(maturities, dtype=np.float64)
    s = np.asarray(spreads, dtype=np.float64)
    if T.shape != s.shape or T.ndim != 1:
        raise ValueError("maturities and spreads must be 1-d arrays of the same length")
    hazards: list[float] = []
    for k in range(T.size):

        def f(h, k=k):
            curve = HazardCurve(T[: k + 1], [*hazards, h])
            return cds_legs(curve, T[k], r, recovery, frequency, steps_per_year).mtm(s[k])

        if f(0.0) > 0:
            raise ValueError(f"spread {s[k]} at {T[k]}y is too low: it implies a negative hazard rate")
        hazards.append(optimize.brentq(f, 0.0, 50.0, xtol=1e-15))
    return HazardCurve(T, hazards)
