"""Breeden-Litzenberger: the risk-neutral density from call prices.

Differentiating :math:`c(K) = e^{-rT} \\int (s - K)^+ f(s)\\, ds` twice in
the strike gives

.. math:: f(K) = e^{rT} \\frac{\\partial^2 c}{\\partial K^2},

so a continuum of call prices determines the market's risk-neutral
distribution of :math:`S_T`. :func:`breeden_litzenberger_density`
estimates it with the three-point second difference, which is exact for
quadratics and allows a non-uniform strike grid; the butterfly spread it
prices is the Arrow-Debreu security of the original paper.

References: D. T. Breeden and R. H. Litzenberger, *J. Business* 51 (1978)
621-651; Hull, *Options, Futures, and Other Derivatives*, 10th ed.,
appendix to Ch. 20.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.options.core.base import DensityResult

__all__ = ["breeden_litzenberger_density"]


def breeden_litzenberger_density(strikes: ArrayLike, call_prices: ArrayLike, T: float, r: float) -> DensityResult:
    """Risk-neutral density of :math:`S_T` from European call prices.

    Parameters
    ----------
    strikes : array_like, shape (n,)
        Strictly increasing strikes, ``n >= 3``.
    call_prices : array_like, shape (n,)
        Call prices at those strikes, same maturity.
    T, r : float
        Maturity and risk-free rate.

    Returns
    -------
    DensityResult
        The density at the ``n - 2`` interior strikes.

    Examples
    --------
    Black-Scholes prices recover the lognormal density:

    >>> import numpy as np
    >>> from scipy.stats import lognorm
    >>> from financialkit.options import black_scholes_price
    >>> K = np.linspace(50.0, 200.0, 601)
    >>> res = breeden_litzenberger_density(K, black_scholes_price(100.0, K, 1.0, 0.05, 0.2), 1.0, 0.05)
    >>> exact = lognorm.pdf(res.strikes, s=0.2, scale=100 * np.exp(0.05 - 0.02))
    >>> bool(np.max(np.abs(res.density - exact)) < 1e-5)
    True
    """
    K = np.asarray(strikes, dtype=np.float64)
    c = np.asarray(call_prices, dtype=np.float64)
    if K.ndim != 1 or K.shape != c.shape or K.size < 3:
        raise ValueError("strikes and call_prices must be 1-d arrays of the same length >= 3")
    if np.any(np.diff(K) <= 0):
        raise ValueError("strikes must be strictly increasing")
    h0, h1 = K[1:-1] - K[:-2], K[2:] - K[1:-1]
    d2c = 2.0 * (h1 * c[:-2] - (h0 + h1) * c[1:-1] + h0 * c[2:]) / (h0 * h1 * (h0 + h1))
    return DensityResult(strikes=K[1:-1], density=np.exp(r * T) * d2c)
