r"""The theta-method time march shared by every one-factor scheme.

With :math:`A` the tridiagonal spatial operator at the interior nodes and
:math:`b` its boundary contributions, one step of size :math:`\Delta\tau`
backward from maturity solves

.. math:: (I - \theta\Delta\tau A) V^{n+1} = (I + (1 - \theta)\Delta\tau A) V^n
          + \Delta\tau\,[\theta b^{n+1} + (1 - \theta) b^n],

which is the explicit scheme for :math:`\theta = 0`, Laasonen's fully
implicit scheme for :math:`\theta = 1` and Crank-Nicolson for
:math:`\theta = 1/2`. Rannacher's start replaces the first few
Crank-Nicolson steps by pairs of implicit half steps, which damps the
high-frequency error that a non-smooth payoff excites. For an American
option each implicit step becomes a linear complementarity problem
against the exercise value, solved by a callable that returns an
:class:`~financialkit.pde_methods.core.base.LCPSolution`; an explicit
step simply takes the larger of the continuation and exercise values.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from financialkit.pde_methods.core.base import LCPSolution
from financialkit.pde_methods.utils.operators import solve_tridiagonal

__all__ = ["theta_march", "LCPSolver", "BoundaryFunc"]

LCPSolver = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], LCPSolution]
"""``solver(lower, diag, upper, rhs, obstacle, x0) -> LCPSolution``."""

BoundaryFunc = Callable[[float], tuple[float, float]]
"""``boundary(tau) -> (V at the lowest node, V at the highest node)``."""


def _step(V, lower, diag, upper, dt, theta, b_new, obstacle, lcp, solver):
    u = V[1:-1]
    rhs = u + (1.0 - theta) * dt * (lower * V[:-2] + diag * u + upper * V[2:])
    out = np.empty_like(V)
    out[0], out[-1] = b_new
    iterations = 0
    if theta == 0.0:
        new = rhs if obstacle is None else np.maximum(rhs, obstacle[1:-1])
    else:
        rhs[0] += theta * dt * lower[0] * b_new[0]
        rhs[-1] += theta * dt * upper[-1] * b_new[1]
        lo, di, up = -theta * dt * lower, 1.0 - theta * dt * diag, -theta * dt * upper
        if obstacle is None:
            new = solve_tridiagonal(lo, di, up, rhs, solver)
        else:
            sol = lcp(lo, di, up, rhs, obstacle[1:-1], np.maximum(u, obstacle[1:-1]))
            new, iterations = sol.x, sol.n_iter
    out[1:-1] = new
    return out, iterations


def theta_march(
    V0: np.ndarray,
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    T: float,
    n_time: int,
    boundary: BoundaryFunc,
    theta: float = 0.5,
    rannacher_steps: int = 0,
    obstacle: np.ndarray | None = None,
    lcp: LCPSolver | None = None,
    solver: str = "banded",
) -> tuple[np.ndarray, int]:
    """March grid values from the payoff (``tau = 0``) to ``tau = T``.

    Parameters
    ----------
    V0 : ndarray, shape (M + 1,)
        Payoff on the grid, boundaries included.
    lower, diag, upper : ndarray, shape (M - 1,)
        Spatial operator at the interior nodes, as returned by
        :func:`~financialkit.pde_methods.utils.operators.black_scholes_operator`.
    T : float
        Time to maturity.
    n_time : int
        Number of full time steps.
    boundary : callable
        ``boundary(tau)`` returns the Dirichlet values at both ends.
    theta : float
        Implicitness in ``[0, 1]``.
    rannacher_steps : int
        Number of initial steps replaced by two implicit half steps each.
    obstacle : ndarray, shape (M + 1,), optional
        Exercise value; given, the option is American.
    lcp : callable, optional
        Linear complementarity solver for implicit American steps.
    solver : {"banded", "thomas"}
        Tridiagonal solver for European implicit steps.

    Returns
    -------
    surface : ndarray, shape (n_time + 1, M + 1)
        ``surface[n]`` holds the values at ``tau = n * T / n_time``.
    lcp_iterations : int
        Total iterations of the complementarity solver.
    """
    if not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta must be in [0, 1], got {theta}")
    if obstacle is not None and theta > 0.0 and lcp is None:
        raise ValueError("an American implicit step needs an lcp solver")
    dt = T / n_time
    surface = np.empty((n_time + 1, V0.size))
    surface[0] = V = np.asarray(V0, dtype=np.float64)
    total = 0
    for n in range(n_time):
        if n < rannacher_steps:
            for half in (0.5, 1.0):
                V, it = _step(V, lower, diag, upper, 0.5 * dt, 1.0, boundary((n + half) * dt), obstacle, lcp, solver)
                total += it
        else:
            V, it = _step(V, lower, diag, upper, dt, theta, boundary((n + 1) * dt), obstacle, lcp, solver)
            total += it
        surface[n + 1] = V
    return surface, total
