"""Extreme value theory: block maxima and peaks over threshold.

Fisher and Tippett (1928) showed that a normalised maximum of i.i.d.
losses can only converge to one of three laws, which Jenkinson and von
Mises unified as the generalized extreme value distribution
:math:`H_\\xi`; fat-tailed losses (Student-t, Pareto) give :math:`\\xi > 0`.
Pickands (1975) and Balkema and de Haan (1974) proved the companion
result for exceedances: above a high threshold :math:`u` the excess
loss is approximately generalized Pareto, :math:`G_{\\xi, \\beta}`,
which yields VaR and ES far beyond the data,

.. math::

   \\mathrm{VaR}_\\alpha = u + \\frac{\\beta}{\\xi}\\left[\\left(\\frac{n}{N_u}(1 - \\alpha)\\right)^{-\\xi} - 1\\right],
   \\qquad \\mathrm{ES}_\\alpha = \\frac{\\mathrm{VaR}_\\alpha + \\beta - \\xi u}{1 - \\xi}.

Fits use :data:`scipy.stats.genextreme` (whose shape is :math:`c = -\\xi`)
and :data:`scipy.stats.genpareto` by maximum likelihood.

References: R. A. Fisher and L. H. C. Tippett, *Proc. Cambridge Phil.
Soc.* 24 (1928) 180-190; J. Pickands, *Ann. Statist.* 3 (1975) 119-131;
A. A. Balkema and L. de Haan, *Ann. Probab.* 2 (1974) 792-804; A. J.
McNeil and R. Frey, *J. Empirical Finance* 7 (2000) 271-300; McNeil,
Frey and Embrechts, *Quantitative Risk Management*, revised ed. (2015),
Ch. 5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from financialkit.risk.core.base import GEVResult, GPDResult

__all__ = ["block_maxima", "fit_gev", "gev_return_level", "mean_excess", "fit_pot", "pot_var", "pot_es"]


def block_maxima(pnl: ArrayLike, block_size: int) -> np.ndarray:
    """Largest loss in each consecutive block (an incomplete last block is dropped).

    Parameters
    ----------
    pnl : array_like, shape (n,)
        Profits and losses in time order.
    block_size : int
        Observations per block (e.g. 21 for months of trading days).

    Returns
    -------
    ndarray, shape (n // block_size,)
    """
    losses = -np.asarray(pnl, dtype=np.float64).ravel()
    m = losses.size // int(block_size)
    if block_size < 1 or m < 1:
        raise ValueError("need at least one full block")
    return losses[: m * block_size].reshape(m, block_size).max(axis=1)


def fit_gev(pnl: ArrayLike, block_size: int) -> GEVResult:
    """Fit the generalized extreme value law to block maxima of the losses.

    Parameters
    ----------
    pnl : array_like, shape (n,)
        Profits and losses in time order.
    block_size : int
        Observations per block.

    Returns
    -------
    GEVResult
    """
    M = block_maxima(pnl, block_size)
    c, loc, scale = stats.genextreme.fit(M)
    return GEVResult(xi=float(-c), mu=float(loc), sigma=float(scale), block_size=int(block_size), maxima=M)


def gev_return_level(fit: GEVResult, k: ArrayLike) -> np.ndarray | float:
    """Loss level exceeded by the block maximum once every ``k`` blocks on average.

    Parameters
    ----------
    fit : GEVResult
    k : array_like
        Return period in blocks (> 1).

    Returns
    -------
    float or ndarray

    Examples
    --------
    For a Gumbel law (:math:`\\xi = 0`) the 100-block level is
    :math:`\\mu - \\sigma \\ln(-\\ln 0.99)`.

    >>> round(gev_return_level(GEVResult(xi=0.0, mu=0.0, sigma=1.0, block_size=21), 100), 4)
    4.6001
    """
    out = stats.genextreme.ppf(1.0 - 1.0 / np.asarray(k, dtype=np.float64), -fit.xi, loc=fit.mu, scale=fit.sigma)
    return float(out) if np.ndim(out) == 0 else out


def mean_excess(pnl: ArrayLike, thresholds: ArrayLike) -> np.ndarray:
    """Empirical mean excess function :math:`e(u) = E[L - u \\mid L > u]` of the losses.

    For a generalized Pareto tail it is linear in :math:`u` with slope
    :math:`\\xi / (1 - \\xi)`, which is how a threshold is chosen: where
    the plot turns straight.

    Parameters
    ----------
    pnl : array_like, shape (n,)
    thresholds : array_like, shape (m,)
        Loss thresholds.

    Returns
    -------
    ndarray, shape (m,)
        ``nan`` where no loss exceeds the threshold.
    """
    L = -np.asarray(pnl, dtype=np.float64).ravel()
    u = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
    out = np.full(u.shape, np.nan)
    for i, ui in enumerate(u):
        ex = L[L > ui] - ui
        if ex.size:
            out[i] = ex.mean()
    return out


def fit_pot(pnl: ArrayLike, threshold: float) -> GPDResult:
    """Fit a generalized Pareto law to the losses above ``threshold``.

    Parameters
    ----------
    pnl : array_like, shape (n,)
        Profits and losses.
    threshold : float
        Loss threshold ``u`` (a positive loss for a loss tail).

    Returns
    -------
    GPDResult
    """
    L = -np.asarray(pnl, dtype=np.float64).ravel()
    excesses = L[L > threshold] - threshold
    if excesses.size < 10:
        raise ValueError(f"only {excesses.size} losses exceed the threshold; at least 10 are needed")
    xi, _, beta = stats.genpareto.fit(excesses, floc=0.0)
    return GPDResult(xi=float(xi), beta=float(beta), threshold=float(threshold), n_exceed=int(excesses.size), n_obs=int(L.size), excesses=excesses)


def pot_var(fit: GPDResult, alpha: ArrayLike = 0.99) -> np.ndarray | float:
    """Peaks-over-threshold VaR (valid for ``1 - alpha`` below the exceedance rate).

    Parameters
    ----------
    fit : GPDResult
    alpha : array_like
        Confidence level.

    Returns
    -------
    float or ndarray
    """
    a = np.asarray(alpha, dtype=np.float64)
    tail = fit.n_obs / fit.n_exceed * (1.0 - a)
    if fit.xi == 0.0:
        out = fit.threshold - fit.beta * np.log(tail)
    else:
        out = fit.threshold + fit.beta / fit.xi * (tail ** (-fit.xi) - 1.0)
    return float(out) if np.ndim(out) == 0 else out


def pot_es(fit: GPDResult, alpha: ArrayLike = 0.99) -> np.ndarray | float:
    """Peaks-over-threshold expected shortfall (finite only for ``xi < 1``).

    Parameters
    ----------
    fit : GPDResult
    alpha : array_like
        Confidence level.

    Returns
    -------
    float or ndarray
    """
    if fit.xi >= 1.0:
        raise ValueError("expected shortfall is infinite for xi >= 1")
    out = (np.asarray(pot_var(fit, alpha)) + fit.beta - fit.xi * fit.threshold) / (1.0 - fit.xi)
    return float(out) if np.ndim(out) == 0 else out
