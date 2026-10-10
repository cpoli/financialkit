"""Copulas: Sklar's decomposition, Gaussian and Student-t copulas, tail dependence.

Sklar (1959) proved that any joint distribution splits into its
marginals and a copula :math:`C`, a distribution on the unit cube with
uniform marginals: :math:`F(x_1, \\dots, x_d) = C(F_1(x_1), \\dots, F_d(x_d))`.
Dependence can therefore be modelled apart from the marginals: sample
the copula, then push each coordinate through any marginal quantile
function. The Gaussian copula has no tail dependence; the t copula
(Demarta and McNeil 2005) has

.. math::

   \\lambda = 2\\, t_{\\nu + 1}\\left(-\\sqrt{\\frac{(\\nu + 1)(1 - \\rho)}{1 + \\rho}}\\right),

so joint crashes stay likely however far into the tail one looks.
Both copulas have Kendall's tau :math:`\\tfrac{2}{\\pi} \\arcsin \\rho`.

References: A. Sklar, *Publ. Inst. Statist. Univ. Paris* 8 (1959)
229-231; S. Demarta and A. J. McNeil, "The t Copula and Related
Copulas," *Int. Statist. Rev.* 73 (2005) 111-129; P. Embrechts, A. J.
McNeil and D. Straumann, "Correlation and Dependence in Risk
Management" (2002).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from financialkit.integrators import RNGLike, correlated_increments

__all__ = [
    "gaussian_copula_sample",
    "t_copula_sample",
    "apply_marginals",
    "t_tail_dependence",
    "empirical_tail_dependence",
    "copula_kendall_tau",
]


def _normals(corr: ArrayLike, n: int, rng: np.random.Generator) -> np.ndarray:
    # One unit step of correlated Brownian increments is a draw from N(0, corr).
    return correlated_increments(corr, T=1.0, n_steps=1, n_paths=n, rng=rng)[:, 0, :]


def gaussian_copula_sample(corr: ArrayLike, n: int, rng: RNGLike = None) -> np.ndarray:
    """Draw from the Gaussian copula with correlation matrix ``corr``.

    Parameters
    ----------
    corr : array_like, shape (d, d)
    n : int
        Number of draws.
    rng : numpy.random.Generator, int or None

    Returns
    -------
    ndarray, shape (n, d)
        Points in the unit cube with uniform marginals.
    """
    return stats.norm.cdf(_normals(corr, n, np.random.default_rng(rng)))


def t_copula_sample(corr: ArrayLike, df: float, n: int, rng: RNGLike = None) -> np.ndarray:
    """Draw from the Student-t copula: correlated normals over a shared :math:`\\sqrt{W / \\nu}`.

    Parameters
    ----------
    corr : array_like, shape (d, d)
    df : float
        Degrees of freedom (> 0); ``inf`` gives the Gaussian copula.
    n : int
    rng : numpy.random.Generator, int or None

    Returns
    -------
    ndarray, shape (n, d)
    """
    rng = np.random.default_rng(rng)
    Z = _normals(corr, n, rng)
    if np.isinf(df):
        return stats.norm.cdf(Z)
    W = rng.chisquare(df, size=(n, 1))
    return stats.t.cdf(Z / np.sqrt(W / df), df)


def apply_marginals(u: ArrayLike, marginals: Sequence[Callable[[np.ndarray], ArrayLike]]) -> np.ndarray:
    """Sklar's construction: give each copula coordinate its own marginal law.

    Parameters
    ----------
    u : array_like, shape (n, d)
        Copula sample.
    marginals : sequence of callables, length d
        Quantile functions, e.g. ``scipy.stats.t(4).ppf``.

    Returns
    -------
    ndarray, shape (n, d)
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != len(marginals):
        raise ValueError("need one marginal quantile function per column of u")
    return np.column_stack([np.asarray(ppf(u[:, j]), dtype=np.float64) for j, ppf in enumerate(marginals)])


def t_tail_dependence(rho: ArrayLike, df: ArrayLike) -> np.ndarray | float:
    """Coefficient of (upper = lower) tail dependence of the bivariate t copula.

    Parameters
    ----------
    rho : array_like
        Correlation parameter.
    df : array_like
        Degrees of freedom; ``inf`` (the Gaussian copula) gives 0 for ``rho < 1``.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Demarta and McNeil (2005), Table 1: :math:`\\nu = 4`, :math:`\\rho = 0.5`.

    >>> round(t_tail_dependence(0.5, 4), 2)
    0.25
    >>> t_tail_dependence(0.5, float("inf"))
    0.0
    """
    rho, df = np.broadcast_arrays(np.asarray(rho, dtype=np.float64), np.asarray(df, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = -np.sqrt((df + 1.0) * (1.0 - rho) / (1.0 + rho))
        out = np.where(np.isinf(df), np.where(rho >= 1.0, 1.0, 0.0), 2.0 * stats.t.cdf(arg, np.where(np.isinf(df), 1.0, df) + 1.0))
    return float(out) if out.ndim == 0 else out


def empirical_tail_dependence(u: ArrayLike, v: ArrayLike, q: ArrayLike = 0.05, tail: str = "lower") -> np.ndarray | float:
    """Finite-level tail dependence :math:`P(V < q \\mid U < q)` (or the upper analogue).

    Parameters
    ----------
    u, v : array_like, shape (n,)
        Copula sample or pseudo-observations.
    q : array_like
        Tail level(s).
    tail : {"lower", "upper"}

    Returns
    -------
    float or ndarray
    """
    u = np.asarray(u, dtype=np.float64)[:, None]
    v = np.asarray(v, dtype=np.float64)[:, None]
    qa = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if tail == "lower":
        joint, marg = (u < qa) & (v < qa), qa
    elif tail == "upper":
        joint, marg = (u > 1.0 - qa) & (v > 1.0 - qa), qa
    else:
        raise ValueError(f"tail must be 'lower' or 'upper', got {tail!r}")
    out = joint.mean(axis=0) / marg
    return float(out[0]) if np.ndim(q) == 0 else out


def copula_kendall_tau(rho: ArrayLike) -> np.ndarray | float:
    """Kendall's tau of a Gaussian or t copula, :math:`\\tfrac{2}{\\pi}\\arcsin\\rho`.

    Parameters
    ----------
    rho : array_like

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(copula_kendall_tau(0.5), 4)
    0.3333
    """
    out = 2.0 / np.pi * np.arcsin(np.asarray(rho, dtype=np.float64))
    return float(out) if out.ndim == 0 else out
