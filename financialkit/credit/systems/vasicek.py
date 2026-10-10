"""Vasicek's one-factor portfolio loss and the large homogeneous portfolio limit.

Vasicek (1987, published 1991 and 2002) let each obligor default when
its asset return :math:`Z_i = \\sqrt\\rho M + \\sqrt{1 - \\rho}\\,\\varepsilon_i`
falls below :math:`N^{-1}(p)`. Given the common factor :math:`M`, defaults
are independent with probability

.. math:: p(M) = N\\left(\\frac{N^{-1}(p) - \\sqrt\\rho\\, M}{\\sqrt{1 - \\rho}}\\right),

so in a large homogeneous portfolio the default fraction converges to
:math:`p(M)`, whose distribution function is

.. math:: F(x) = N\\left(\\frac{\\sqrt{1 - \\rho}\\, N^{-1}(x) - N^{-1}(p)}{\\sqrt\\rho}\\right).

Its quantiles are what the Basel II IRB formula charges capital for.

References: O. Vasicek, "Probability of Loss on Loan Portfolio" (KMV,
1987); O. Vasicek, "Limiting Loan Loss Probability Distribution" (KMV,
1991); O. Vasicek, "The Distribution of Loan Portfolio Value," *Risk*
15 (December 2002) 160-162.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from financialkit.credit.core.base import LossDistribution
from financialkit.credit.utils.numerics import normal_factor_quadrature
from financialkit.integrators import RNGLike
from financialkit.options.utils.common import as_output

__all__ = [
    "conditional_default_probability",
    "vasicek_cdf",
    "vasicek_pdf",
    "vasicek_quantile",
    "simulate_one_factor_defaults",
    "one_factor_loss_distribution",
]


def conditional_default_probability(p: ArrayLike, rho: ArrayLike, M: ArrayLike) -> np.ndarray | float:
    """Default probability given the common factor, :math:`p(M)`.

    Parameters
    ----------
    p : array_like
        Unconditional default probability.
    rho : array_like
        Asset correlation in ``[0, 1)``.
    M : array_like
        Factor value.

    Returns
    -------
    float or ndarray
    """
    p, rho, M = (np.asarray(a, dtype=np.float64) for a in (p, rho, M))
    return as_output(stats.norm.cdf((stats.norm.ppf(p) - np.sqrt(rho) * M) / np.sqrt(1.0 - rho)))


def vasicek_cdf(x: ArrayLike, p: ArrayLike, rho: ArrayLike) -> np.ndarray | float:
    """Distribution function of the large-portfolio default fraction.

    Parameters
    ----------
    x : array_like
        Default fraction in ``(0, 1)``.
    p, rho : array_like

    Returns
    -------
    float or ndarray

    Examples
    --------
    The median default rate lies below the mean ``p`` (the law is right-skewed):

    >>> vasicek_cdf(0.02, 0.02, 0.2) > 0.5
    True
    """
    x, p, rho = (np.asarray(a, dtype=np.float64) for a in (x, p, rho))
    return as_output(stats.norm.cdf((np.sqrt(1.0 - rho) * stats.norm.ppf(x) - stats.norm.ppf(p)) / np.sqrt(rho)))


def vasicek_pdf(x: ArrayLike, p: ArrayLike, rho: ArrayLike) -> np.ndarray | float:
    """Density of the large-portfolio default fraction.

    Parameters
    ----------
    x, p, rho : array_like

    Returns
    -------
    float or ndarray
    """
    x, p, rho = (np.asarray(a, dtype=np.float64) for a in (x, p, rho))
    zx = stats.norm.ppf(x)
    arg = (np.sqrt(1.0 - rho) * zx - stats.norm.ppf(p)) / np.sqrt(rho)
    return as_output(np.sqrt((1.0 - rho) / rho) * stats.norm.pdf(arg) / stats.norm.pdf(zx))


def vasicek_quantile(alpha: ArrayLike, p: ArrayLike, rho: ArrayLike) -> np.ndarray | float:
    """Quantile of the large-portfolio default fraction (the worst-case default rate).

    .. math:: N\\left(\\frac{N^{-1}(p) + \\sqrt\\rho\\, N^{-1}(\\alpha)}{\\sqrt{1 - \\rho}}\\right)

    Parameters
    ----------
    alpha : array_like
        Confidence level, e.g. ``0.999``.
    p, rho : array_like

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> x = vasicek_quantile(0.999, 0.01, 0.2)
    >>> round(vasicek_cdf(x, 0.01, 0.2), 10)
    0.999
    """
    a, p, rho = (np.asarray(v, dtype=np.float64) for v in (alpha, p, rho))
    return as_output(stats.norm.cdf((stats.norm.ppf(p) + np.sqrt(rho) * stats.norm.ppf(a)) / np.sqrt(1.0 - rho)))


def simulate_one_factor_defaults(p: ArrayLike, rho: float, n_obligors: int, n_sims: int, rng: RNGLike = None) -> np.ndarray:
    """Number of defaults per scenario in a homogeneous one-factor portfolio.

    Draws the factor, then the conditionally independent defaults as a
    binomial count.

    Parameters
    ----------
    p : array_like
        Default probability (scalar for a homogeneous pool).
    rho : float
        Asset correlation.
    n_obligors, n_sims : int
    rng : numpy.random.Generator, int or None

    Returns
    -------
    ndarray of int, shape (n_sims,)
    """
    rng = np.random.default_rng(rng)
    pM = np.asarray(conditional_default_probability(p, rho, rng.standard_normal(n_sims)))
    return rng.binomial(n_obligors, pM)


def one_factor_loss_distribution(p: float, rho: float, n_obligors: int, lgd: float = 1.0, n_nodes: int = 128) -> LossDistribution:
    """Exact loss distribution of a finite homogeneous portfolio, by factor quadrature.

    :math:`P(K = k) = E_M[\\binom{n}{k} p(M)^k (1 - p(M))^{n - k}]`, the
    expectation over :math:`M` computed with Gauss-Hermite quadrature.

    Parameters
    ----------
    p, rho : float
    n_obligors : int
    lgd : float
        Loss given default per obligor, as a fraction of its exposure.
    n_nodes : int
        Quadrature nodes.

    Returns
    -------
    LossDistribution
        Losses as fractions of the portfolio notional.
    """
    x, w = normal_factor_quadrature(n_nodes)
    pM = np.asarray(conditional_default_probability(p, rho, x))
    k = np.arange(n_obligors + 1)
    pmf = w @ stats.binom.pmf(k[None, :], n_obligors, pM[:, None])
    return LossDistribution(losses=lgd * k / n_obligors, pmf=pmf, method="one_factor_binomial")
