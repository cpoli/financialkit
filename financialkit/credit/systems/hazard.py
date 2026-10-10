"""Reduced-form default: Jarrow and Turnbull's hazard rates and survival curves.

Jarrow and Turnbull (1995) stopped modelling why a firm defaults and
modelled when: default is the first jump of a Poisson process with
intensity :math:`\\lambda`, so survival to :math:`t` is
:math:`Q(t) = e^{-\\int_0^t \\lambda(s)\\,ds}`. With the intensity piecewise
constant between knots this is the curve every CDS desk bootstraps. With
recovery of treasury (a fraction :math:`\\delta` of a riskless bond is paid
at default), a defaultable zero-coupon bond is worth

.. math:: P^d(0, T) = P(0, T)\\,[\\delta + (1 - \\delta) Q(T)],

and for small default rates the spread obeys the credit triangle
:math:`s \\approx \\lambda (1 - R)`.

References: R. A. Jarrow and S. M. Turnbull, "Pricing Derivatives on
Financial Securities Subject to Credit Risk," *J. Finance* 50 (1995)
53-85; D. Lando, "On Cox Processes and Credit Risky Securities," *Rev.
Derivatives Res.* 2 (1998) 99-120.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.credit.core.base import SurvivalCurve
from financialkit.options.utils.common import as_output

__all__ = ["HazardCurve", "jarrow_turnbull_zero", "hazard_from_spread", "spread_from_hazard"]


class HazardCurve(SurvivalCurve):
    """Piecewise-constant default intensity.

    The hazard is ``hazards[i]`` on ``(knots[i-1], knots[i]]`` (with
    ``knots[-1] = 0``) and stays at the last value beyond the last knot.

    Parameters
    ----------
    knots : array_like, shape (n,)
        Increasing end times of the hazard segments (years, positive).
    hazards : array_like, shape (n,)
        Non-negative intensities.

    Examples
    --------
    >>> c = HazardCurve([1.0, 3.0], [0.01, 0.03])
    >>> round(c.survival(2.0), 6) == round(float(np.exp(-0.04)), 6)
    True
    >>> round(c.inverse_cumulative_hazard(0.04), 12)
    2.0
    """

    def __init__(self, knots: ArrayLike, hazards: ArrayLike):
        knots = np.atleast_1d(np.asarray(knots, dtype=np.float64))
        hazards = np.atleast_1d(np.asarray(hazards, dtype=np.float64))
        if knots.shape != hazards.shape or knots.ndim != 1:
            raise ValueError("knots and hazards must be 1-d arrays of the same length")
        if knots[0] <= 0 or np.any(np.diff(knots) <= 0):
            raise ValueError("knots must be positive and strictly increasing")
        if np.any(hazards < 0):
            raise ValueError("hazard rates must be non-negative")
        self.knots, self.hazards = knots, hazards
        starts = np.r_[0.0, knots[:-1]]
        self._starts = starts
        self._cum = np.r_[0.0, np.cumsum(hazards * (knots - starts))]

    @classmethod
    def flat(cls, hazard: float) -> HazardCurve:
        """A constant intensity: exponential default time with mean ``1 / hazard``."""
        return cls([1.0], [hazard])

    def _segment(self, t: np.ndarray) -> np.ndarray:
        return np.minimum(np.searchsorted(self.knots, t, side="left"), self.knots.size - 1)

    def hazard(self, t: ArrayLike) -> np.ndarray | float:
        """Intensity :math:`\\lambda(t)`."""
        t = np.asarray(t, dtype=np.float64)
        return as_output(self.hazards[self._segment(t)])

    def cumulative_hazard(self, t: ArrayLike) -> np.ndarray | float:
        """:math:`\\Lambda(t)`, linear between knots."""
        t = np.maximum(np.asarray(t, dtype=np.float64), 0.0)
        i = self._segment(t)
        return as_output(self._cum[i] + self.hazards[i] * (t - self._starts[i]))

    def inverse_cumulative_hazard(self, x: ArrayLike) -> np.ndarray | float:
        """Time at which :math:`\\Lambda` reaches ``x`` (``inf`` if it never does)."""
        x = np.asarray(x, dtype=np.float64)
        i = np.clip(np.searchsorted(self._cum, x, side="left") - 1, 0, self.knots.size - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = self._starts[i] + (x - self._cum[i]) / self.hazards[i]
        t = np.where((self.hazards[i] == 0) & (x > self._cum[i]), np.inf, t)
        return as_output(np.where(x <= 0, 0.0, t))

    def __repr__(self) -> str:
        return f"HazardCurve(knots={self.knots.tolist()!r}, hazards={self.hazards.tolist()!r})"


def jarrow_turnbull_zero(T: ArrayLike, r: ArrayLike, curve: SurvivalCurve, recovery: float = 0.0) -> np.ndarray | float:
    """Defaultable zero-coupon bond under recovery of treasury.

    .. math:: P^d(0, T) = e^{-rT}\\,[\\delta + (1 - \\delta) Q(T)]

    Parameters
    ----------
    T : array_like
        Maturities.
    r : array_like
        Flat risk-free rate.
    curve : SurvivalCurve
        Risk-neutral survival curve.
    recovery : float
        Fraction :math:`\\delta` of the riskless bond paid at default.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(jarrow_turnbull_zero(5.0, 0.05, HazardCurve.flat(0.02), 0.4), 6)
    0.734333
    """
    T, r = np.asarray(T, dtype=np.float64), np.asarray(r, dtype=np.float64)
    Q = np.asarray(curve.survival(T), dtype=np.float64)
    return as_output(np.exp(-r * T) * (recovery + (1.0 - recovery) * Q))


def hazard_from_spread(spread: ArrayLike, recovery: float = 0.4) -> np.ndarray | float:
    """Credit triangle: the hazard rate implied by a spread, :math:`\\lambda = s / (1 - R)`.

    Exact for a continuously paid CDS premium and a flat hazard and rate.

    Parameters
    ----------
    spread : array_like
    recovery : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(hazard_from_spread(0.012, 0.4), 6)
    0.02
    """
    return as_output(np.asarray(spread, dtype=np.float64) / (1.0 - recovery))


def spread_from_hazard(hazard: ArrayLike, recovery: float = 0.4) -> np.ndarray | float:
    """Inverse credit triangle, :math:`s = \\lambda (1 - R)`.

    Parameters
    ----------
    hazard : array_like
    recovery : float

    Returns
    -------
    float or ndarray
    """
    return as_output(np.asarray(hazard, dtype=np.float64) * (1.0 - recovery))
