"""Compiled tridiagonal kernels shared by the finite-difference schemes.

Every kernel takes a tridiagonal matrix as three full-length arrays:
``lower[i]`` multiplies ``x[i - 1]`` (``lower[0]`` is ignored),
``diag[i]`` multiplies ``x[i]`` and ``upper[i]`` multiplies ``x[i + 1]``
(``upper[-1]`` is ignored). Scalar settings travel in a ``params`` array,
so one compiled kernel serves every grid. Without Numba they run as plain
Python loops with identical results.
"""

from __future__ import annotations

import numpy as np

from financialkit._jit import njit

__all__ = ["thomas_kernel", "psor_kernel", "brennan_schwartz_kernel"]


@njit
def thomas_kernel(lower, diag, upper, rhs):
    """Solve a tridiagonal system by forward elimination and back substitution."""
    m = diag.shape[0]
    c = np.empty(m)
    d = np.empty(m)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, m):
        denom = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denom
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom
    x = np.empty(m)
    x[m - 1] = d[m - 1]
    for i in range(m - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


@njit
def psor_kernel(lower, diag, upper, rhs, obstacle, x0, params):
    """Projected SOR for ``A x >= rhs, x >= obstacle``; ``params = [omega, tol, max_iter]``.

    Returns the solution and the number of sweeps (``max_iter + 1`` if it did not converge).
    """
    omega, tol, max_iter = params[0], params[1], int(params[2])
    m = diag.shape[0]
    x = x0.copy()
    for it in range(1, max_iter + 1):
        err = 0.0
        for i in range(m):
            s = rhs[i]
            if i > 0:
                s -= lower[i] * x[i - 1]
            if i < m - 1:
                s -= upper[i] * x[i + 1]
            y = x[i] + omega * (s / diag[i] - x[i])
            if y < obstacle[i]:
                y = obstacle[i]
            err = max(err, abs(y - x[i]))
            x[i] = y
        if err < tol:
            return x, it
    return x, max_iter + 1


@njit
def brennan_schwartz_kernel(lower, diag, upper, rhs, obstacle, params):
    """Direct LCP solve; ``params = [side]`` with ``side = -1`` when exercise happens at low indices (a put), ``+1`` at high ones (a call)."""
    m = diag.shape[0]
    b = diag.copy()
    r = rhs.copy()
    x = np.empty(m)
    if params[0] < 0:
        # eliminate the upper diagonal from the top index down, then substitute upward with projection
        for i in range(m - 2, -1, -1):
            f = upper[i] / b[i + 1]
            b[i] -= f * lower[i + 1]
            r[i] -= f * r[i + 1]
        x[0] = max(r[0] / b[0], obstacle[0])
        for i in range(1, m):
            x[i] = max((r[i] - lower[i] * x[i - 1]) / b[i], obstacle[i])
    else:
        for i in range(1, m):
            f = lower[i] / b[i - 1]
            b[i] -= f * upper[i - 1]
            r[i] -= f * r[i - 1]
        x[m - 1] = max(r[m - 1] / b[m - 1], obstacle[m - 1])
        for i in range(m - 2, -1, -1):
            x[i] = max((r[i] - upper[i] * x[i + 1]) / b[i], obstacle[i])
    return x
