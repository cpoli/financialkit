r"""Richardson extrapolation of grid-dependent prices.

If a scheme's error behaves like :math:`V_h = V + C h^p + o(h^p)`, two
solves on grids :math:`h` and :math:`h/\rho` cancel the leading term:

.. math:: V \approx V_{h/\rho} + \frac{V_{h/\rho} - V_h}{\rho^p - 1}.

Lewis Fry Richardson (1911) introduced this "deferred approach to the
limit" for finite-difference stresses in masonry dams. Three solves also
estimate the order actually achieved,
:math:`\hat p = \log_\rho\bigl[(V_{h} - V_{h/\rho})/(V_{h/\rho} - V_{h/\rho^2})\bigr]`,
a check that the error really is a clean power of :math:`h` (it is for
Crank-Nicolson with the strike on a node and Rannacher's start; it is not
for a raw Crank-Nicolson solve with an oscillating error).

References: L. F. Richardson, *Phil. Trans. Roy. Soc. A* 210 (1911)
307-357; L. F. Richardson and J. A. Gaunt, *Phil. Trans. Roy. Soc. A* 226
(1927) 299-361.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.pde_methods.core.base import RichardsonResult

__all__ = ["richardson_extrapolation"]


def richardson_extrapolation(values: ArrayLike, ratio: float = 2.0, order: float | None = None) -> RichardsonResult:
    """Extrapolate approximations on successively refined grids to zero step size.

    Parameters
    ----------
    values : array_like, shape (k,)
        Approximations on grids refined by `ratio` each time, coarsest first
        (at least two).
    ratio : float
        Refinement ratio between consecutive grids.
    order : float, optional
        Error order :math:`p`. If omitted, it is estimated from the three
        finest values when there are three or more, and taken as 2
        otherwise.

    Returns
    -------
    RichardsonResult

    Examples
    --------
    Values with error :math:`h^2` extrapolate exactly:

    >>> res = richardson_extrapolation([1.0 + 0.04, 1.0 + 0.01, 1.0 + 0.0025])
    >>> round(res.extrapolated, 12), round(res.observed_order, 12)
    (1.0, 2.0)
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 2:
        raise ValueError("need at least two approximations")
    if ratio <= 1.0:
        raise ValueError("ratio must exceed 1")
    observed = np.nan
    if v.size >= 3:
        d1, d2 = v[-2] - v[-3], v[-1] - v[-2]
        if d1 != 0.0 and d2 != 0.0 and d1 / d2 > 0:
            observed = float(np.log(d1 / d2) / np.log(ratio))
    p = float(order) if order is not None else (observed if np.isfinite(observed) else 2.0)
    extrapolated = float(v[-1] + (v[-1] - v[-2]) / (ratio**p - 1.0))
    return RichardsonResult(values=v, extrapolated=extrapolated, order=p, ratio=float(ratio), observed_order=observed)
