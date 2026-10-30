r"""Spatial operators, tridiagonal solves and grid interpolation for
financialkit.pde_methods.

On a grid :math:`S_0 < \dots < S_M` with local spacings
:math:`h_- = S_i - S_{i-1}` and :math:`h_+ = S_{i+1} - S_i`, the
Black-Scholes operator

.. math:: \mathcal{L}V = \tfrac12 \sigma^2 S^2 V_{SS} + (r - q) S V_S - r V

is discretised with the three-point central differences

.. math:: V_S \approx \frac{-h_+^2 V_{i-1} + (h_+^2 - h_-^2) V_i + h_-^2 V_{i+1}}{h_- h_+ (h_- + h_+)}, \qquad
          V_{SS} \approx \frac{2 (h_+ V_{i-1} - (h_- + h_+) V_i + h_- V_{i+1})}{h_- h_+ (h_- + h_+)},

second-order accurate on smoothly varying grids and the familiar
formulas when :math:`h_- = h_+`. See D. Tavella and C. Randall, *Pricing
Financial Instruments: The Finite Difference Method* (Wiley, 2000),
Ch. 5, and P. Wilmott, *Paul Wilmott on Quantitative Finance*, 2nd ed.
(Wiley, 2006), Ch. 77-78.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded

from financialkit.options.utils.common import option_sign
from financialkit.pde_methods.core.kernels import thomas_kernel

__all__ = [
    "black_scholes_operator",
    "tridiagonal_matrix",
    "solve_tridiagonal",
    "discounted_intrinsic",
    "interpolate_at",
]


def black_scholes_operator(S: ArrayLike, sigma: ArrayLike, r: float, q: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tridiagonal coefficients of the Black-Scholes operator at the interior nodes.

    Parameters
    ----------
    S : array_like, shape (M + 1,)
        Strictly increasing spot grid, boundaries included.
    sigma : float or array_like, shape (M - 1,)
        Volatility, or a local volatility at each interior node.
    r, q : float
        Risk-free rate and dividend yield.

    Returns
    -------
    lower, diag, upper : ndarray, shape (M - 1,)
        ``(LV)_i = lower[i] V_{i} + diag[i] V_{i+1} + upper[i] V_{i+2}`` for
        interior node ``i + 1``; ``lower[0]`` and ``upper[-1]`` multiply the
        boundary values.

    Examples
    --------
    On a uniform grid the coefficients are Wilmott's
    :math:`\\tfrac12(\\sigma^2 i^2 \\mp (r - q) i)` and :math:`-(\\sigma^2 i^2 + r)`:

    >>> lo, di, up = black_scholes_operator([0.0, 1.0, 2.0], 0.2, 0.05)
    >>> [round(float(c[0]), 6) for c in (lo, di, up)]
    [-0.005, -0.09, 0.045]
    """
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 1 or S.size < 3 or np.any(np.diff(S) <= 0):
        raise ValueError("S must be a strictly increasing grid of at least three nodes")
    hm, hp = np.diff(S)[:-1], np.diff(S)[1:]
    s = S[1:-1]
    diffusion, drift = 0.5 * np.asarray(sigma, dtype=np.float64) ** 2 * s**2, (r - q) * s
    denom = hm * hp * (hm + hp)
    lower = (2.0 * diffusion * hp - drift * hp**2) / denom
    upper = (2.0 * diffusion * hm + drift * hm**2) / denom
    diag = (-2.0 * diffusion * (hm + hp) + drift * (hp**2 - hm**2)) / denom - r
    return lower, diag, upper


def tridiagonal_matrix(lower: ArrayLike, diag: ArrayLike, upper: ArrayLike) -> sp.csr_matrix:
    """Assemble the sparse matrix of a tridiagonal system in the kernels' full-length convention.

    Parameters
    ----------
    lower, diag, upper : array_like, shape (m,)
        ``lower[0]`` and ``upper[-1]`` are ignored.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (m, m)
    """
    lower, diag, upper = (np.asarray(a, dtype=np.float64) for a in (lower, diag, upper))
    return sp.diags([lower[1:], diag, upper[:-1]], [-1, 0, 1], format="csr")


def solve_tridiagonal(lower: ArrayLike, diag: ArrayLike, upper: ArrayLike, rhs: ArrayLike, solver: str = "banded") -> np.ndarray:
    """Solve a tridiagonal system with :func:`scipy.linalg.solve_banded` or the Thomas kernel.

    Parameters
    ----------
    lower, diag, upper : array_like, shape (m,)
        ``lower[0]`` and ``upper[-1]`` are ignored.
    rhs : array_like, shape (m,) or (m, k)
        Right-hand side(s); several columns need ``solver="banded"``.
    solver : {"banded", "thomas"}

    Returns
    -------
    ndarray, shape like `rhs`
    """
    lower, diag, upper, rhs = (np.asarray(a, dtype=np.float64) for a in (lower, diag, upper, rhs))
    if solver == "thomas":
        if rhs.ndim != 1:
            raise ValueError('solver="thomas" takes a single right-hand side')
        return thomas_kernel(lower, diag, upper, rhs)
    if solver != "banded":
        raise ValueError(f'solver must be "banded" or "thomas", got {solver!r}')
    ab = np.zeros((3, diag.size))
    ab[0, 1:], ab[1], ab[2, :-1] = upper[:-1], diag, lower[1:]
    return solve_banded((1, 1), ab, rhs, check_finite=False)


def discounted_intrinsic(S: ArrayLike, K: float, tau: float, r: float, q: float = 0.0, option_type: str = "call") -> np.ndarray:
    r"""Value of a vanilla option far from the strike: :math:`\max(\pm(S e^{-q\tau} - K e^{-r\tau}), 0)`.

    This is the Black-Scholes price in the limits :math:`S \to 0` and
    :math:`S \to \infty`, and serves as the Dirichlet boundary condition.

    Parameters
    ----------
    S : array_like
    K, tau, r, q : float
        Strike, time to maturity, rate and dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    ndarray
    """
    sign = option_sign(option_type)
    S = np.asarray(S, dtype=np.float64)
    return np.maximum(sign * (S * np.exp(-q * tau) - K * np.exp(-r * tau)), 0.0)


def interpolate_at(S_grid: ArrayLike, V: ArrayLike, S: float) -> tuple[float, float, float]:
    """Value, delta and gamma at `S` from a cubic spline through the grid values.

    Parameters
    ----------
    S_grid, V : array_like, shape (M + 1,)
    S : float
        Spot inside the grid.

    Returns
    -------
    price, delta, gamma : float
    """
    S_grid, V = np.asarray(S_grid, dtype=np.float64), np.asarray(V, dtype=np.float64)
    if not S_grid[0] <= S <= S_grid[-1]:
        raise ValueError(f"spot {S} outside the grid [{S_grid[0]}, {S_grid[-1]}]")
    spline = CubicSpline(S_grid, V)
    return float(spline(S)), float(spline(S, 1)), float(spline(S, 2))
