"""Small numerical helpers shared by the portfolio optimizers.

Long-only problems have no closed form, so they go to
:func:`scipy.optimize.minimize` with SLSQP and exact gradients; the
helpers here keep that call, and the moment arithmetic around it, in one
place.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from financialkit.constants import DEFAULT_MAX_ITER

__all__ = ["as_mean_cov", "portfolio_moments", "solve_qp"]


def as_mean_cov(mu: Optional[ArrayLike], cov: ArrayLike) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Validate and convert a mean vector and covariance matrix.

    Parameters
    ----------
    mu : array_like, shape (d,), or None
    cov : array_like, shape (d, d)

    Returns
    -------
    mu : ndarray or None
    cov : ndarray
    """
    C = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"cov must be a square matrix, got shape {C.shape}")
    if mu is None:
        return None, C
    m = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    if m.shape != (C.shape[0],):
        raise ValueError(f"mu must have shape ({C.shape[0]},), got {m.shape}")
    return m, C


def portfolio_moments(weights: ArrayLike, mu: Optional[ArrayLike], cov: ArrayLike) -> tuple[float, float]:
    """Expected return and volatility of a portfolio.

    Parameters
    ----------
    weights : array_like, shape (d,)
    mu : array_like, shape (d,), or None
        Expected returns; the mean is ``nan`` without them.
    cov : array_like, shape (d, d)

    Returns
    -------
    mean, volatility : float

    Examples
    --------
    >>> portfolio_moments([0.5, 0.5], [0.1, 0.2], [[0.04, 0.0], [0.0, 0.09]])
    (0.15000000000000002, 0.18027756377319948)
    """
    w = np.asarray(weights, dtype=np.float64)
    m, C = as_mean_cov(mu, cov)
    mean = float(w @ m) if m is not None else float("nan")
    return mean, float(np.sqrt(w @ C @ w))


def solve_qp(
    Q: ArrayLike,
    A_eq: ArrayLike,
    b_eq: ArrayLike,
    A_ineq: Optional[ArrayLike] = None,
    b_ineq: Optional[ArrayLike] = None,
    bounds: tuple[float, float] = (0.0, 1.0),
    c: Optional[ArrayLike] = None,
    x0: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Minimize :math:`\\tfrac12 x'Qx - c'x` subject to linear constraints and bounds.

    Parameters
    ----------
    Q : array_like, shape (d, d)
        Positive semi-definite matrix.
    A_eq, b_eq : array_like, shapes (k, d) and (k,)
        Equality constraints ``A_eq @ x == b_eq``.
    A_ineq, b_ineq : array_like, optional
        Inequality constraints ``A_ineq @ x >= b_ineq``.
    bounds : tuple of float
        Lower and upper bound on every component.
    c : array_like, shape (d,), optional
        Linear term.
    x0 : array_like, shape (d,), optional
        Starting point; equal weights by default.

    Returns
    -------
    ndarray, shape (d,)

    Examples
    --------
    Long-only minimum variance with one asset that would be shorted
    unconstrained:

    >>> import numpy as np
    >>> cov = np.array([[0.04, 0.05], [0.05, 0.09]])
    >>> np.round(solve_qp(cov, np.ones((1, 2)), [1.0]), 6)
    array([1., 0.])
    """
    Q = np.asarray(Q, dtype=np.float64)
    d = Q.shape[0]
    lin = np.zeros(d) if c is None else np.asarray(c, dtype=np.float64)
    Ae, be = np.atleast_2d(np.asarray(A_eq, dtype=np.float64)), np.atleast_1d(np.asarray(b_eq, dtype=np.float64))
    cons = [{"type": "eq", "fun": lambda x: Ae @ x - be, "jac": lambda x: Ae}]
    if A_ineq is not None:
        Ai, bi = np.atleast_2d(np.asarray(A_ineq, dtype=np.float64)), np.atleast_1d(np.asarray(b_ineq, dtype=np.float64))
        cons.append({"type": "ineq", "fun": lambda x: Ai @ x - bi, "jac": lambda x: Ai})
    start = np.full(d, 1.0 / d) if x0 is None else np.asarray(x0, dtype=np.float64)
    res = optimize.minimize(
        lambda x: 0.5 * x @ Q @ x - lin @ x,
        start,
        jac=lambda x: Q @ x - lin,
        method="SLSQP",
        bounds=[bounds] * d,
        constraints=cons,
        options={"ftol": 1e-15, "maxiter": DEFAULT_MAX_ITER},
    )
    if not res.success:
        raise RuntimeError(f"quadratic program failed: {res.message}")
    return np.asarray(res.x)
