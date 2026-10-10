"""Li's Gaussian copula for correlated default times.

Li (2000) gave each name its own survival curve, read off its CDS
quotes, and joined the default times with a Gaussian copula: draw
correlated normals, map them to uniforms :math:`U_i`, and set
:math:`\\tau_i = Q_i^{-1}(1 - U_i)`. Every marginal is exactly the
market-implied one, and the joint probability that two names default
within :math:`T` is

.. math:: P(\\tau_1 \\le T, \\tau_2 \\le T) = \\Phi_2\\big(N^{-1}(p_1), N^{-1}(p_2); \\rho\\big),

with :math:`p_i = P(\\tau_i \\le T)`. The model priced basket default
swaps and CDO tranches for a decade; its thin joint tail (see
:func:`financialkit.risk.t_tail_dependence`) is the reason it was
blamed when they failed in 2007-2008.

References: D. X. Li, "On Default Correlation: A Copula Function
Approach," *J. Fixed Income* 9 (2000) 43-54.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from financialkit.credit.core.base import SurvivalCurve
from financialkit.integrators import RNGLike
from financialkit.risk import gaussian_copula_sample, t_copula_sample

__all__ = ["copula_default_times", "joint_default_probability", "default_correlation", "nth_to_default_probability"]


def _corr(rho: float | ArrayLike, d: int) -> np.ndarray:
    R = np.asarray(rho, dtype=np.float64)
    if R.ndim == 0:
        R = np.full((d, d), float(R))
        np.fill_diagonal(R, 1.0)
    return R


def copula_default_times(
    curves: SurvivalCurve | Sequence[SurvivalCurve], rho: float | ArrayLike, n: int, n_names: int | None = None, df: float | None = None, rng: RNGLike = None
) -> np.ndarray:
    """Simulate correlated default times with a Gaussian (or t) copula.

    Parameters
    ----------
    curves : SurvivalCurve or sequence of SurvivalCurve
        One curve per name, or one curve shared by ``n_names`` names.
    rho : float or array_like, shape (d, d)
        Common pairwise correlation or a full correlation matrix.
    n : int
        Number of scenarios.
    n_names : int, optional
        Number of names when a single curve is given.
    df : float, optional
        Degrees of freedom of a t copula; Gaussian if omitted.
    rng : numpy.random.Generator, int or None

    Returns
    -------
    ndarray, shape (n, d)
        Default times (``inf`` for a name with zero hazard that never defaults).
    """
    if isinstance(curves, SurvivalCurve):
        if n_names is None:
            raise ValueError("n_names is required with a single curve")
        curve_list = [curves] * n_names
    else:
        curve_list = list(curves)
    d = len(curve_list)
    R = _corr(rho, d)
    U = gaussian_copula_sample(R, n, rng) if df is None else t_copula_sample(R, df, n, rng)
    return np.column_stack([c.sample_default_times(n, uniforms=1.0 - U[:, j]) for j, c in enumerate(curve_list)])


def joint_default_probability(p1: ArrayLike, p2: ArrayLike, rho: ArrayLike) -> np.ndarray | float:
    """Probability that both names default by the horizon under a Gaussian copula.

    Parameters
    ----------
    p1, p2 : array_like
        Marginal default probabilities to the horizon.
    rho : array_like
        Copula correlation.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(joint_default_probability(0.05, 0.05, 0.0), 6)
    0.0025
    """
    p1, p2, rho = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (p1, p2, rho)))
    out = np.empty(p1.shape)
    for idx in np.ndindex(p1.shape):
        r = float(rho[idx])
        out[idx] = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]], allow_singular=True).cdf(
            [stats.norm.ppf(p1[idx]), stats.norm.ppf(p2[idx])]
        )
    return float(out) if out.ndim == 0 else out


def default_correlation(p1: ArrayLike, p2: ArrayLike, rho: ArrayLike) -> np.ndarray | float:
    """Correlation of the two default indicators implied by copula correlation ``rho``.

    Parameters
    ----------
    p1, p2, rho : array_like

    Returns
    -------
    float or ndarray
    """
    p12 = np.asarray(joint_default_probability(p1, p2, rho))
    p1, p2 = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
    out = (p12 - p1 * p2) / np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    return float(out) if out.ndim == 0 else out


def nth_to_default_probability(times: ArrayLike, k: int, T: ArrayLike) -> np.ndarray | float:
    """Probability that at least ``k`` names default by ``T``, from simulated default times.

    Parameters
    ----------
    times : array_like, shape (n, d)
    k : int
        ``1`` for first-to-default.
    T : array_like
        Horizon(s).

    Returns
    -------
    float or ndarray
    """
    times = np.asarray(times, dtype=np.float64)
    kth = np.sort(times, axis=1)[:, k - 1]
    Ts = np.asarray(T, dtype=np.float64)
    out = (kth[:, None] <= np.atleast_1d(Ts)[None, :]).mean(axis=0)
    return float(out[0]) if Ts.ndim == 0 else out
