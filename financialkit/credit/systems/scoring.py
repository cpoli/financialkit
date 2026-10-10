"""Altman's Z-score: discriminant analysis of accounting ratios.

Altman (1968) ran a linear discriminant analysis on 33 bankrupt and 33
surviving US manufacturers and found that five ratios separate them:

.. math:: Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 1.0 X_5,

with :math:`X_1` working capital, :math:`X_2` retained earnings,
:math:`X_3` earnings before interest and taxes and :math:`X_5` sales, all
over total assets, and :math:`X_4` the market value of equity over the
book value of total liabilities. Firms below 1.81 fell in the distress
zone, above 2.99 in the safe zone, and the band between was a zone of
ignorance. Fisher's linear discriminant (:func:`linear_discriminant`)
re-derives such weights from any labelled sample.

References: E. I. Altman, "Financial Ratios, Discriminant Analysis and
the Prediction of Corporate Bankruptcy," *J. Finance* 23 (1968) 589-609;
R. A. Fisher, "The Use of Multiple Measurements in Taxonomic Problems,"
*Ann. Eugenics* 7 (1936) 179-188.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.options.utils.common import as_output

__all__ = ["ALTMAN_COEFFICIENTS", "altman_ratios", "altman_z_score", "altman_zone", "linear_discriminant"]

ALTMAN_COEFFICIENTS = np.array([1.2, 1.4, 3.3, 0.6, 1.0])
"""Altman's (1968) weights on :math:`X_1, \\dots, X_5`."""


def altman_ratios(
    working_capital: ArrayLike,
    retained_earnings: ArrayLike,
    ebit: ArrayLike,
    market_equity: ArrayLike,
    sales: ArrayLike,
    total_assets: ArrayLike,
    total_liabilities: ArrayLike,
) -> np.ndarray:
    """Altman's five ratios :math:`X_1, \\dots, X_5`, stacked on the last axis.

    Parameters
    ----------
    working_capital, retained_earnings, ebit, market_equity, sales, total_assets, total_liabilities : array_like
        Balance-sheet and income-statement items, in the same currency unit.

    Returns
    -------
    ndarray, shape (..., 5)
    """
    wc, re, eb, me, s, ta, tl = (
        np.asarray(a, dtype=np.float64) for a in (working_capital, retained_earnings, ebit, market_equity, sales, total_assets, total_liabilities)
    )
    return np.stack(np.broadcast_arrays(wc / ta, re / ta, eb / ta, me / tl, s / ta), axis=-1)


def altman_z_score(
    working_capital: ArrayLike,
    retained_earnings: ArrayLike,
    ebit: ArrayLike,
    market_equity: ArrayLike,
    sales: ArrayLike,
    total_assets: ArrayLike,
    total_liabilities: ArrayLike,
) -> np.ndarray | float:
    """Altman's (1968) Z-score for public manufacturers.

    Parameters
    ----------
    working_capital, retained_earnings, ebit, market_equity, sales, total_assets, total_liabilities : array_like

    Returns
    -------
    float or ndarray

    Examples
    --------
    A firm with total assets 670, working capital 170, retained earnings
    300, EBIT 60, sales 2200, equity worth 380 and liabilities of 240:

    >>> round(altman_z_score(170, 300, 60, 380, 2200, 670, 240), 2)
    5.46
    """
    return as_output(altman_ratios(working_capital, retained_earnings, ebit, market_equity, sales, total_assets, total_liabilities) @ ALTMAN_COEFFICIENTS)


def altman_zone(z: ArrayLike, lower: float = 1.81, upper: float = 2.99) -> np.ndarray | str:
    """Classify Z-scores into Altman's distress, grey and safe zones.

    Parameters
    ----------
    z : array_like
    lower, upper : float
        Zone boundaries (Altman 1968).

    Returns
    -------
    str or ndarray of str

    Examples
    --------
    >>> altman_zone([1.2, 2.5, 3.4]).tolist()
    ['distress', 'grey', 'safe']
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.where(z < lower, "distress", np.where(z > upper, "safe", "grey"))
    return str(out) if out.ndim == 0 else out


def linear_discriminant(x_good: ArrayLike, x_bad: ArrayLike) -> tuple[np.ndarray, float]:
    """Fisher's linear discriminant between two groups of ratio vectors.

    The weights :math:`w = S_W^{-1}(\\bar x_g - \\bar x_b)` maximise the
    separation of the group means relative to the pooled within-group
    covariance :math:`S_W`; the cutoff is the midpoint of the projected
    means (equal priors). Weights are only defined up to scale.

    Parameters
    ----------
    x_good, x_bad : array_like, shape (n_g, d) and (n_b, d)
        Ratios of surviving and failed firms.

    Returns
    -------
    weights : ndarray, shape (d,)
    cutoff : float
        Scores above the cutoff are classified as survivors.
    """
    g = np.asarray(x_good, dtype=np.float64)
    b = np.asarray(x_bad, dtype=np.float64)
    ng, nb = g.shape[0], b.shape[0]
    Sw = ((ng - 1) * np.cov(g, rowvar=False) + (nb - 1) * np.cov(b, rowvar=False)) / (ng + nb - 2)
    w = np.linalg.solve(np.atleast_2d(Sw), g.mean(axis=0) - b.mean(axis=0))
    cutoff = 0.5 * float(w @ (g.mean(axis=0) + b.mean(axis=0)))
    return w, cutoff
