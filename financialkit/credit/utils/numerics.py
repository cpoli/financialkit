"""Numerics behind the portfolio-loss models: Panjer's recursion and factor quadrature.

Panjer (1981) showed that when the claim count :math:`N` satisfies
:math:`P(N = n) = (a + b/n) P(N = n - 1)` (Poisson, binomial and negative
binomial do) and severities are integers, the aggregate-loss
probabilities obey

.. math:: f_n = \\frac{1}{1 - a s_0} \\sum_{j=1}^{n} \\left(a + \\frac{b j}{n}\\right) s_j f_{n-j},

an :math:`O(n^2)` recursion with no Fourier transform or simulation.
:func:`normal_factor_quadrature` gives Gauss-Hermite nodes and weights
for expectations over a standard-normal common factor.

References: H. H. Panjer, "Recursive Evaluation of a Family of Compound
Distributions," *ASTIN Bull.* 12 (1981) 22-26.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.typing import ArrayLike

__all__ = ["panjer_recursion", "normal_factor_quadrature"]


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
