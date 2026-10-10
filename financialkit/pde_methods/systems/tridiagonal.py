"""The Thomas algorithm for tridiagonal systems.

Every implicit finite-difference step in one space dimension is a
tridiagonal system. Gaussian elimination specialised to that structure
needs no pivoting when the matrix is diagonally dominant, which the
implicit Black-Scholes matrix is, and costs :math:`O(m)` instead of
:math:`O(m^3)` operations: one sweep eliminates the sub-diagonal, a
second substitutes back. The library schemes call
:func:`scipy.linalg.solve_banded` by default (LAPACK's ``gtsv``-type
banded solver); this hand-written version is the lesson, and the two
agree to rounding.

References: L. H. Thomas, "Elliptic Problems in Linear Difference
Equations over a Network," Watson Scientific Computing Laboratory
report, Columbia University (1949); G. H. Golub and C. F. Van Loan,
*Matrix Computations*, 4th ed. (Johns Hopkins, 2013), Section 4.3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.pde_methods.core.kernels import thomas_kernel

__all__ = ["thomas_solve"]


def thomas_solve(lower: ArrayLike, diag: ArrayLike, upper: ArrayLike, rhs: ArrayLike) -> np.ndarray:
    """Solve the tridiagonal system ``A x = rhs`` by the Thomas algorithm.

    Parameters
    ----------
    lower : array_like, shape (m,)
        Sub-diagonal, ``lower[i]`` multiplying ``x[i - 1]``; ``lower[0]`` is ignored.
    diag : array_like, shape (m,)
        Main diagonal.
    upper : array_like, shape (m,)
        Super-diagonal, ``upper[i]`` multiplying ``x[i + 1]``; ``upper[-1]`` is ignored.
    rhs : array_like, shape (m,)

    Returns
    -------
    ndarray, shape (m,)

    Examples
    --------
    The second-difference matrix ``tridiag(-1, 2, -1)``:

    >>> x = thomas_solve([0, -1, -1], [2, 2, 2], [-1, -1, 0], [1, 0, 1])
    >>> x.round(12).tolist()
    [1.0, 1.0, 1.0]
    """
    lower, diag, upper, rhs = (np.ascontiguousarray(a, dtype=np.float64) for a in (lower, diag, upper, rhs))
    if not lower.shape == diag.shape == upper.shape == rhs.shape or diag.ndim != 1:
        raise ValueError("lower, diag, upper and rhs must be 1-D arrays of one length")
    return thomas_kernel(lower, diag, upper, rhs)
