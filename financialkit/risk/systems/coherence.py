"""Coherent risk measures (Artzner, Delbaen, Eber and Heath 1999).

A risk measure :math:`\\rho` maps a random profit and loss to the capital
that makes the position acceptable. It is *coherent* when it is

- monotone: :math:`Z \\le X` in every state implies :math:`\\rho(Z) \\ge \\rho(X)`;
- subadditive: :math:`\\rho(X + Y) \\le \\rho(X) + \\rho(Y)`, so merging
  never creates risk;
- positively homogeneous: :math:`\\rho(\\lambda X) = \\lambda \\rho(X)`;
- translation invariant: :math:`\\rho(X + c) = \\rho(X) - c`.

VaR is all of these but subadditive, which fails for skewed positions
such as defaultable bonds; expected shortfall satisfies all four.

References: P. Artzner, F. Delbaen, J.-M. Eber and D. Heath, "Coherent
Measures of Risk," *Mathematical Finance* 9 (1999) 203-228.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.constants import DEFAULT_ATOL, DEFAULT_RTOL
from financialkit.risk.core.base import CoherenceResult, RiskMeasure
from financialkit.risk.utils.tail import weighted_tail

__all__ = ["ValueAtRisk", "ExpectedShortfall", "check_coherence"]


class ValueAtRisk(RiskMeasure):
    """VaR as a :class:`~financialkit.risk.core.base.RiskMeasure` on equally likely scenarios.

    Examples
    --------
    >>> ValueAtRisk(0.95)([-5.0, -1.0, 0.0, 1.0, 2.0] * 4)
    5.0
    """

    name = "VaR"

    def __call__(self, pnl: ArrayLike) -> float:
        return weighted_tail(pnl, self.alpha)[0]


class ExpectedShortfall(RiskMeasure):
    """Acerbi-Tasche expected shortfall as a :class:`~financialkit.risk.core.base.RiskMeasure`."""

    name = "ES"

    def __call__(self, pnl: ArrayLike) -> float:
        return weighted_tail(pnl, self.alpha)[1]


def check_coherence(measure: RiskMeasure, X: ArrayLike, Y: ArrayLike, lam: float = 2.0, c: float = 1.0) -> CoherenceResult:
    """Test the four coherence axioms of ``measure`` on two positions.

    Parameters
    ----------
    measure : RiskMeasure
    X, Y : array_like, shape (n,)
        Profits and losses of two positions in the same ``n`` equally
        likely states of the world.
    lam : float
        Positive scale for the homogeneity check.
    c : float
        Cash amount for the translation check.

    Returns
    -------
    CoherenceResult
        A failed axiom is a counterexample; passing on one pair proves nothing.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape != Y.shape:
        raise ValueError("X and Y must be scenario vectors over the same states")
    rx, ry, rs = measure(X), measure(Y), measure(X + Y)
    tol = DEFAULT_ATOL + DEFAULT_RTOL * max(abs(rx), abs(ry), 1.0)
    return CoherenceResult(
        monotonicity=bool(measure(np.minimum(X, Y)) >= max(rx, ry) - tol),
        subadditivity=bool(rs <= rx + ry + tol),
        positive_homogeneity=bool(abs(measure(lam * X) - lam * rx) <= lam * tol),
        translation_invariance=bool(abs(measure(X + c) - (rx - c)) <= tol + DEFAULT_RTOL * abs(c)),
        rho_x=rx,
        rho_y=ry,
        rho_sum=rs,
    )
