"""The Fundamental Review of the Trading Book: 97.5% expected shortfall
scaled to each risk factor's liquidity horizon.

The Basel Committee's 2016 market-risk standard replaced the 99% VaR by
a 97.5% expected shortfall over a base horizon of :math:`T = 10` days,
and stopped pretending every position can be closed in ten days. Each
risk factor gets a liquidity horizon of 10, 20, 40, 60 or 120 days, and
ES is computed once with every factor shocked and again with only the
factors at least as illiquid as each horizon, then cascaded:

.. math::

   \\mathrm{ES} = \\sqrt{\\mathrm{ES}_T(P)^2 + \\sum_{j \\ge 2}
   \\left(\\mathrm{ES}_T(P, j) \\sqrt{\\frac{LH_j - LH_{j-1}}{T}}\\right)^2}.

A factor with horizon :math:`LH_j` thus has its ten-day shocks scaled up
by :math:`\\sqrt{LH_j / T}` overall, but only the increments beyond the
shorter horizons are added in quadrature.

References: Basel Committee on Banking Supervision, *Minimum Capital
Requirements for Market Risk* (January 2016), paragraphs 181-182 and
Table 2; Basel Committee, *Fundamental Review of the Trading Book: A
Revised Market Risk Framework*, consultative document (October 2013).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from financialkit.risk.core.base import FRTBResult
from financialkit.risk.systems.expected_shortfall import expected_shortfall
from financialkit.risk.utils.tail import check_alpha

__all__ = ["LIQUIDITY_HORIZONS", "frtb_es_cascade", "frtb_expected_shortfall"]

LIQUIDITY_HORIZONS = (10, 20, 40, 60, 120)
"""The FRTB liquidity horizons in days (Basel 2016, paragraph 181(k))."""

_BASE_HORIZON = 10


def _weights(horizons: np.ndarray) -> np.ndarray:
    return np.sqrt(np.diff(horizons, prepend=0) / _BASE_HORIZON)


def frtb_es_cascade(es_by_horizon: ArrayLike) -> float:
    """Liquidity-adjusted ES from the five ten-day ES figures of the FRTB cascade.

    Parameters
    ----------
    es_by_horizon : array_like, shape (5,)
        :math:`\\mathrm{ES}_T(P, j)` for :math:`LH_j` = 10, 20, 40, 60 and
        120 days: the ten-day ES with only the risk factors whose
        liquidity horizon is at least :math:`LH_j` shocked (all of them
        for the first).

    Returns
    -------
    float

    Examples
    --------
    If every factor has a 120-day horizon, all five figures are equal
    and the cascade is the square-root-of-time rule,
    :math:`\\sqrt{120/10} \\approx 3.46`.

    >>> round(frtb_es_cascade([1.0] * 5), 6)
    3.464102
    """
    e = np.asarray(es_by_horizon, dtype=np.float64).ravel()
    if e.size != len(LIQUIDITY_HORIZONS):
        raise ValueError(f"es_by_horizon needs one value per liquidity horizon {LIQUIDITY_HORIZONS}, got {e.size}")
    return float(np.linalg.norm(e * _weights(np.asarray(LIQUIDITY_HORIZONS, dtype=np.float64))))


def frtb_expected_shortfall(
    pnl_fn: Callable[[np.ndarray], ArrayLike],
    scenarios: ArrayLike,
    liquidity_horizons: ArrayLike,
    alpha: float = 0.975,
) -> FRTBResult:
    """FRTB liquidity-adjusted expected shortfall of a book from ten-day scenarios.

    For each horizon :math:`LH_j`, the moves of factors with a shorter
    horizon are set to zero, the book is revalued with ``pnl_fn`` and the
    Acerbi-Tasche ES is taken with
    :func:`~financialkit.risk.systems.expected_shortfall.expected_shortfall`;
    the five figures are combined by :func:`frtb_es_cascade`.

    Parameters
    ----------
    pnl_fn : callable
        ``pnl_fn(dX) -> pnl`` maps ten-day factor moves of shape
        ``(n, d)`` to the book's profit and loss, shape ``(n,)``.
    scenarios : array_like, shape (n, d)
        Ten-day risk-factor moves, simulated or historical (in the
        standard, calibrated to a period of stress).
    liquidity_horizons : array_like of int, shape (d,)
        Each factor's horizon, one of 10, 20, 40, 60 or 120 days.
    alpha : float
        Confidence level; the standard sets 0.975.

    Returns
    -------
    FRTBResult
    """
    alpha = check_alpha(alpha)
    dX = np.atleast_2d(np.asarray(scenarios, dtype=np.float64))
    lh = np.asarray(liquidity_horizons).ravel()
    if lh.size != dX.shape[1]:
        raise ValueError(f"liquidity_horizons needs one entry per factor ({dX.shape[1]}), got {lh.size}")
    if not np.all(np.isin(lh, LIQUIDITY_HORIZONS)):
        raise ValueError(f"liquidity horizons must be in {LIQUIDITY_HORIZONS}, got {np.unique(lh).tolist()}")
    es = np.empty(len(LIQUIDITY_HORIZONS))
    for j, h in enumerate(LIQUIDITY_HORIZONS):
        pnl = np.asarray(pnl_fn(dX * (lh >= h)), dtype=np.float64).ravel()
        if pnl.size != dX.shape[0]:
            raise ValueError(f"pnl_fn must return one value per scenario ({dX.shape[0]}), got {pnl.size}")
        es[j] = expected_shortfall(pnl, alpha)
    horizons = np.asarray(LIQUIDITY_HORIZONS, dtype=np.float64)
    return FRTBResult(es=frtb_es_cascade(es), es_by_horizon=es, horizons=horizons, scaled=es * _weights(horizons), alpha=alpha)
