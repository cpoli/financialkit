"""CreditRisk+: actuarial portfolio losses by Panjer's recursion.

Credit Suisse First Boston (1997) borrowed the insurer's toolkit. Each
obligor defaults as a Poisson event with rate :math:`p_i`, exposures are
rounded to integer multiples :math:`\\nu_i` of a loss unit, and default
rates share a gamma-distributed factor with mean one and standard
deviation :math:`\\sigma`. The default count is then negative binomial,
which belongs to Panjer's (a, b, 0) class with

.. math:: a = \\delta, \\quad b = (\\alpha - 1)\\delta, \\quad
          \\alpha = \\sigma^{-2}, \\quad \\delta = \\frac{\\mu\\sigma^2}{1 + \\mu\\sigma^2},

where :math:`\\mu = \\sum_i p_i`, so the whole loss distribution follows
from :func:`~financialkit.credit.utils.numerics.panjer_recursion` with
the exposure-band severities :math:`s_j = \\sum_{\\nu_i = j} p_i / \\mu`.
With :math:`\\sigma = 0` the count is Poisson and losses are compound
Poisson.

References: Credit Suisse First Boston, *CreditRisk+: A Credit Risk
Management Framework* (1997), Sections A4-A10; H. H. Panjer, *ASTIN
Bull.* 12 (1981) 22-26.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.credit.core.base import LossDistribution
from financialkit.credit.utils.numerics import panjer_recursion

__all__ = ["creditrisk_plus"]


def creditrisk_plus(exposures: ArrayLike, pds: ArrayLike, sigma: float = 0.0, unit: float = 1.0, n_max: int | None = None) -> LossDistribution:
    """Single-sector CreditRisk+ loss distribution.

    Parameters
    ----------
    exposures : array_like, shape (m,)
        Loss given default of each obligor, in currency.
    pds : array_like, shape (m,)
        Default probabilities (used as Poisson rates).
    sigma : float
        Relative standard deviation of the sector default rate; ``0`` for
        independent (Poisson) defaults.
    unit : float
        Loss unit; exposures are rounded to multiples of it (at least one unit).
    n_max : int, optional
        Largest loss, in units; defaults to the mean plus 15 standard deviations.

    Returns
    -------
    LossDistribution
        Losses in currency, on the grid ``unit * arange(n_max + 1)``.

    Examples
    --------
    The expected loss is the sum of rounded exposures times default probabilities:

    >>> res = creditrisk_plus([100.0, 200.0, 300.0], [0.01, 0.02, 0.03], sigma=0.5, unit=100.0)
    >>> round(res.mean, 4)
    14.0
    """
    E = np.asarray(exposures, dtype=np.float64)
    p = np.asarray(pds, dtype=np.float64)
    if E.shape != p.shape:
        raise ValueError("exposures and pds must have the same shape")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    nu = np.maximum(np.rint(E / unit), 1).astype(np.intp)
    mu = float(p.sum())
    mean = float(p @ nu)
    var = float(p @ nu**2) + sigma**2 * mean**2
    if n_max is None:
        n_max = int(np.ceil(mean + 15.0 * np.sqrt(var))) + int(nu.max())
    severity = np.bincount(nu, weights=p, minlength=int(nu.max()) + 1) / mu
    if sigma == 0.0:
        a, b, f0 = 0.0, mu, float(np.exp(-mu))
    else:
        alpha = 1.0 / sigma**2
        delta = mu * sigma**2 / (1.0 + mu * sigma**2)
        a, b, f0 = delta, (alpha - 1.0) * delta, float((1.0 - delta) ** alpha)
    pmf = panjer_recursion(severity, a, b, f0, n_max)
    return LossDistribution(losses=unit * np.arange(n_max + 1), pmf=pmf, method="creditrisk+")
