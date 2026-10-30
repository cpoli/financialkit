"""Numerics behind the portfolio-loss models: Panjer's and Andersen-Sidenius-Basu's recursions and factor quadrature.

Panjer (1981) showed that when the claim count :math:`N` satisfies
:math:`P(N = n) = (a + b/n) P(N = n - 1)` (Poisson, binomial and negative
binomial do) and severities are integers, the aggregate-loss
probabilities obey

.. math:: f_n = \\frac{1}{1 - a s_0} \\sum_{j=1}^{n} \\left(a + \\frac{b j}{n}\\right) s_j f_{n-j},

an :math:`O(n^2)` recursion with no Fourier transform or simulation.
:func:`loss_recursion` builds the loss distribution of independent
names with integer losses one name at a time (Andersen, Sidenius and
Basu 2003), and :func:`normal_factor_quadrature` gives Gauss-Hermite nodes
and weights for expectations over a standard-normal common factor.

References: H. H. Panjer, "Recursive Evaluation of a Family of Compound
Distributions," *ASTIN Bull.* 12 (1981) 22-26; L. Andersen, J. Sidenius
and S. Basu, "All Your Hedges in One Basket," *Risk* 16(11) (2003) 67-72.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.typing import ArrayLike

__all__ = ["panjer_recursion", "loss_recursion", "normal_factor_quadrature"]


def panjer_recursion(severity: ArrayLike, a: float, b: float, f0: float, n_max: int) -> np.ndarray:
    """Aggregate-loss probabilities of an (a, b, 0) claim count with integer severities.

    Parameters
    ----------
    severity : array_like, shape (m,)
        ``severity[j]`` is the probability that one claim has size ``j``.
    a, b : float
        Panjer class parameters: Poisson ``(0, mu)``, negative binomial
        ``(p, (alpha - 1) p)`` with success probability ``1 - p``.
    f0 : float
        Probability of a zero aggregate loss.
    n_max : int
        Largest aggregate loss computed.

    Returns
    -------
    ndarray, shape (n_max + 1,)

    Examples
    --------
    A Poisson count with unit claims is just the Poisson law:

    >>> from scipy import stats
    >>> f = panjer_recursion([0.0, 1.0], 0.0, 2.0, float(np.exp(-2.0)), 10)
    >>> np.allclose(f, stats.poisson.pmf(np.arange(11), 2.0))
    True
    """
    s = np.zeros(n_max + 1)
    sev = np.asarray(severity, dtype=np.float64)[: n_max + 1]
    s[: sev.size] = sev
    f = np.zeros(n_max + 1)
    f[0] = f0
    j = np.arange(1, n_max + 1)
    c = 1.0 / (1.0 - a * s[0])
    for n in range(1, n_max + 1):
        jj = j[:n]
        f[n] = c * np.sum((a + b * jj / n) * s[jj] * f[n - jj])
    return f


def loss_recursion(p: ArrayLike, loss_units: ArrayLike) -> np.ndarray:
    """Loss distribution of independent defaults with integer losses, added one name at a time.

    With :math:`f^{(k)}` the distribution of the loss of the first
    :math:`k` names, adding name :math:`k + 1`, which defaults with
    probability :math:`p` and then loses :math:`u` units, gives

    .. math:: f^{(k+1)}_l = (1 - p) f^{(k)}_l + p f^{(k)}_{l - u},

    which is exact, costs :math:`O(n L)` and needs no assumption that the
    names are alike (Andersen, Sidenius and Basu 2003). Every row of `p`
    is one scenario, such as one value of a common factor.

    Parameters
    ----------
    p : array_like, shape (n_names,) or (n_scenarios, n_names)
        Default probabilities, independent within a row.
    loss_units : array_like of int, shape (n_names,)
        Loss of each name on default, in units.

    Returns
    -------
    ndarray, shape (sum(loss_units) + 1,) or (n_scenarios, sum(loss_units) + 1)
        Probability of each total loss ``0, 1, ..., sum(loss_units)``.

    Examples
    --------
    Alike names with unit losses give the binomial law:

    >>> from scipy import stats
    >>> f = loss_recursion(np.full(10, 0.2), np.ones(10, dtype=int))
    >>> np.allclose(f, stats.binom.pmf(np.arange(11), 10, 0.2))
    True
    """
    p = np.asarray(p, dtype=np.float64)
    units = np.asarray(loss_units)
    if units.ndim != 1 or units.size != p.shape[-1]:
        raise ValueError("loss_units needs one entry per name")
    if np.any(units < 0) or not np.all(units == np.round(units)):
        raise ValueError("loss_units must be non-negative integers")
    units = units.astype(np.int64)
    P = np.atleast_2d(p)
    f = np.zeros((P.shape[0], int(units.sum()) + 1))
    f[:, 0] = 1.0
    top = 0
    for k, u in enumerate(units):
        pk = P[:, k : k + 1]
        shifted = f[:, : top + 1].copy()
        f[:, : top + 1] *= 1.0 - pk
        f[:, u : top + u + 1] += pk * shifted
        top += u
    return f[0] if p.ndim == 1 else f


def normal_factor_quadrature(n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and weights for :math:`E[g(M)]`, :math:`M \\sim N(0, 1)`.

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    nodes, weights : ndarray, shape (n,)
        Weights sum to one.

    Examples
    --------
    >>> x, w = normal_factor_quadrature(20)
    >>> round(float(w @ x**4), 10)
    3.0
    """
    x, w = hermegauss(n)
    return x, w / w.sum()
