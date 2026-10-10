r"""Linear complementarity solvers for American options.

An implicit step of an American option asks for grid values :math:`x`
that solve the linear complementarity problem (LCP)

.. math:: A x \ge b, \qquad x \ge g, \qquad (A x - b)^\top (x - g) = 0,

with :math:`A` the implicit tridiagonal matrix, :math:`b` the explicit
right-hand side and :math:`g` the exercise value: at every node the
option either continues (the PDE holds) or is exercised (it equals its
payoff). Three classical answers:

* Cryer's projected successive over-relaxation (1971): Gauss-Seidel
  sweeps with over-relaxation :math:`\omega \in (1, 2)`, each update
  projected onto :math:`x \ge g`. Simple and general, but the number of
  sweeps grows with the grid.
* Brennan and Schwartz (1977): one Gaussian elimination in the right
  direction, projecting during back substitution. Exact in one pass when
  the exercise region is a single interval at one end of the grid, as it
  is for a vanilla put or call.
* Forsyth and Vetzal's penalty method (2002): replace the constraint by a
  large penalty :math:`\rho` on nodes that violate it and solve the
  resulting nonlinear system by Newton's (policy) iteration, which
  converges in a handful of tridiagonal solves.

References: C. W. Cryer, *SIAM J. Control* 9 (1971) 385-392; M. J. Brennan
and E. S. Schwartz, *J. Finance* 32 (1977) 449-462; P. A. Forsyth and
K. R. Vetzal, *SIAM J. Sci. Comput.* 23 (2002) 2095-2122; P. Wilmott,
J. Dewynne and S. Howison, *Option Pricing: Mathematical Models and
Computation* (Oxford Financial Press, 1993), Ch. 9.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.constants import DEFAULT_MAX_ITER
from financialkit.pde_methods.core.base import LCPSolution
from financialkit.pde_methods.core.kernels import brennan_schwartz_kernel, psor_kernel
from financialkit.pde_methods.utils.operators import solve_tridiagonal

__all__ = ["psor", "brennan_schwartz", "penalty_solve"]


def _arrays(*arrays: ArrayLike) -> list[np.ndarray]:
    out = [np.ascontiguousarray(a, dtype=np.float64) for a in arrays]
    if any(a.shape != out[0].shape or a.ndim != 1 for a in out):
        raise ValueError("all inputs must be 1-D arrays of one length")
    return out


def psor(
    lower: ArrayLike,
    diag: ArrayLike,
    upper: ArrayLike,
    rhs: ArrayLike,
    obstacle: ArrayLike,
    x0: ArrayLike | None = None,
    omega: float = 1.2,
    tol: float = 1e-8,
    max_iter: int = DEFAULT_MAX_ITER,
) -> LCPSolution:
    """Cryer's projected SOR for a tridiagonal LCP.

    Parameters
    ----------
    lower, diag, upper : array_like, shape (m,)
        Tridiagonal matrix (``lower[0]`` and ``upper[-1]`` ignored).
    rhs : array_like, shape (m,)
    obstacle : array_like, shape (m,)
        Lower bound :math:`g` on the solution.
    x0 : array_like, shape (m,), optional
        Starting point (default: the obstacle).
    omega : float
        Over-relaxation parameter in ``(0, 2)``.
    tol : float
        Stop when no component moves by more than `tol` in a sweep.
    max_iter : int

    Returns
    -------
    LCPSolution

    Examples
    --------
    With a low obstacle the constraint is inactive and PSOR solves ``A x = b``:

    >>> sol = psor([0, -1, -1], [2, 2, 2], [-1, -1, 0], [1, 0, 1], [-5, -5, -5])
    >>> sol.x.round(6).tolist()
    [1.0, 1.0, 1.0]
    """
    if not 0.0 < omega < 2.0:
        raise ValueError(f"omega must be in (0, 2), got {omega}")
    lower, diag, upper, rhs, obstacle = _arrays(lower, diag, upper, rhs, obstacle)
    start = obstacle.copy() if x0 is None else np.maximum(np.asarray(x0, dtype=np.float64), obstacle)
    x, it = psor_kernel(lower, diag, upper, rhs, obstacle, start, np.array([omega, tol, max_iter], dtype=np.float64))
    if it > max_iter:
        raise RuntimeError(f"PSOR did not converge in {max_iter} sweeps")
    return LCPSolution(x=x, iterations=int(it), method="psor")


def brennan_schwartz(lower: ArrayLike, diag: ArrayLike, upper: ArrayLike, rhs: ArrayLike, obstacle: ArrayLike, exercise: str = "low") -> LCPSolution:
    """Brennan and Schwartz's direct solve of a tridiagonal LCP.

    Parameters
    ----------
    lower, diag, upper : array_like, shape (m,)
        Tridiagonal matrix (``lower[0]`` and ``upper[-1]`` ignored).
    rhs, obstacle : array_like, shape (m,)
    exercise : {"low", "high"}
        End of the grid where the exercise region lies: ``"low"`` for a
        put, ``"high"`` for a call.

    Returns
    -------
    LCPSolution

    Examples
    --------
    >>> sol = brennan_schwartz([0, -1, -1], [2, 2, 2], [-1, -1, 0], [1, 0, 1], [1.5, 0, 0])
    >>> sol.x.round(6).tolist()
    [1.5, 1.333333, 1.166667]
    """
    if exercise not in ("low", "high"):
        raise ValueError(f'exercise must be "low" or "high", got {exercise!r}')
    lower, diag, upper, rhs, obstacle = _arrays(lower, diag, upper, rhs, obstacle)
    x = brennan_schwartz_kernel(lower, diag, upper, rhs, obstacle, np.array([-1.0 if exercise == "low" else 1.0]))
    return LCPSolution(x=x, iterations=1, method="brennan_schwartz")


def penalty_solve(
    lower: ArrayLike,
    diag: ArrayLike,
    upper: ArrayLike,
    rhs: ArrayLike,
    obstacle: ArrayLike,
    x0: ArrayLike | None = None,
    penalty: float = 1e8,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> LCPSolution:
    r"""Forsyth and Vetzal's penalty iteration for a tridiagonal LCP.

    Iterates :math:`(A + P^k) x^{k+1} = b + P^k g`, where :math:`P^k` is
    diagonal with :math:`\rho` on the nodes where :math:`x^k < g` and 0
    elsewhere, until the set of penalised nodes stops changing. The
    solution satisfies the constraint to within :math:`O(1/\rho)`.

    Parameters
    ----------
    lower, diag, upper : array_like, shape (m,)
        Tridiagonal matrix (``lower[0]`` and ``upper[-1]`` ignored).
    rhs, obstacle : array_like, shape (m,)
    x0 : array_like, shape (m,), optional
        Starting point (default: the unconstrained solution).
    penalty : float
        Penalty factor :math:`\rho`.
    tol : float
        Relative change below which the iteration stops.
    max_iter : int

    Returns
    -------
    LCPSolution

    Examples
    --------
    >>> sol = penalty_solve([0, -1, -1], [2, 2, 2], [-1, -1, 0], [1, 0, 1], [1.5, 0, 0])
    >>> sol.x.round(6).tolist()
    [1.5, 1.333333, 1.166667]
    """
    lower, diag, upper, rhs, obstacle = _arrays(lower, diag, upper, rhs, obstacle)
    x = solve_tridiagonal(lower, diag, upper, rhs) if x0 is None else np.asarray(x0, dtype=np.float64)
    for it in range(1, max_iter + 1):
        p = np.where(x < obstacle, penalty, 0.0)
        x_new = solve_tridiagonal(lower, diag + p, upper, rhs + p * obstacle)
        p_new = np.where(x_new < obstacle, penalty, 0.0)
        change = np.max(np.abs(x_new - x) / np.maximum(1.0, np.abs(x_new)))
        x = x_new
        if np.array_equal(p_new, p) or change < tol:
            return LCPSolution(x=x, iterations=it, method="penalty")
    raise RuntimeError(f"penalty iteration did not converge in {max_iter} iterations")
