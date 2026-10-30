"""Portfolio loss distributions by recursion: Andersen, Sidenius and Basu (2003).

Vasicek's formula and the binomial mixture behind Hull-White tranche
pricing need a homogeneous pool: every name with the same default
probability, correlation and exposure. Real CDO and CDS-index pools are
not like that. Andersen, Sidenius and Basu kept the one-factor copula
and observed that, given the factor :math:`M`, defaults are independent,
so the conditional loss distribution can be built exactly one name at a
time,

.. math:: f^{(k+1)}_l(M) = (1 - p_{k+1}(M)) f^{(k)}_l(M) + p_{k+1}(M) f^{(k)}_{l - u_{k+1}}(M),

on a grid of loss units, with :math:`p_k(M)` Vasicek's conditional
default probability of name :math:`k` and :math:`u_k` its loss given
default in units. The unconditional distribution is a Gauss-Hermite
average over :math:`M`. The cost is linear in the number of names and
in the number of loss levels, and the result is exact for any mixture
of probabilities, correlations and exposures, with no simulation.

References: L. Andersen, J. Sidenius and S. Basu, "All Your Hedges in One
Basket," *Risk* 16(11) (2003) 67-72; D. O'Kane, *Modelling Single-name
and Multi-name Credit Derivatives* (Wiley, 2008), Ch. 13.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.credit.core.base import LossDistribution
from financialkit.credit.systems.vasicek import conditional_default_probability
from financialkit.credit.utils.numerics import loss_recursion, normal_factor_quadrature

__all__ = ["asb_loss_distribution"]


def asb_loss_distribution(p: ArrayLike, rho: ArrayLike, loss_units: ArrayLike | None = None, unit: float = 1.0, n_nodes: int = 64) -> LossDistribution:
    """Loss distribution of a heterogeneous one-factor portfolio by the ASB recursion.

    Parameters
    ----------
    p : array_like, shape (n_names,)
        Default probability of each name to the horizon.
    rho : array_like
        Asset correlation of each name with the common factor (its loading
        squared), one value for all or one per name.
    loss_units : array_like of int, shape (n_names,), optional
        Loss of each name on default in units of `unit`; one unit each by default.
    unit : float
        Size of a loss unit, e.g. a fraction of the portfolio notional.
    n_nodes : int
        Gauss-Hermite nodes for the factor.

    Returns
    -------
    LossDistribution
        Losses ``unit * k`` for ``k = 0, ..., sum(loss_units)``.

    Examples
    --------
    Alike names reproduce the homogeneous binomial mixture:

    >>> from financialkit.credit import one_factor_loss_distribution
    >>> asb = asb_loss_distribution(np.full(50, 0.02), 0.3, unit=0.6 / 50)
    >>> np.allclose(asb.pmf, one_factor_loss_distribution(0.02, 0.3, 50, lgd=0.6, n_nodes=64).pmf)
    True
    """
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if p.ndim != 1:
        raise ValueError("p must be one-dimensional")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("default probabilities must lie in [0, 1]")
    rho = np.broadcast_to(np.asarray(rho, dtype=np.float64), p.shape)
    if np.any((rho < 0) | (rho >= 1)):
        raise ValueError("rho must lie in [0, 1)")
    units = np.ones(p.size, dtype=np.int64) if loss_units is None else np.asarray(loss_units)
    x, w = normal_factor_quadrature(n_nodes)
    pM = np.asarray(conditional_default_probability(p[None, :], rho[None, :], x[:, None]), dtype=np.float64)
    pmf = w @ loss_recursion(pM, units)
    return LossDistribution(losses=unit * np.arange(pmf.size), pmf=pmf, method="asb_recursion")
