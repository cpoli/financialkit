r"""Spot grids for the finite-difference schemes.

A uniform grid wastes nodes far from the strike, where the option value
is nearly linear, and has too few where the payoff kink makes the
solution curve. Tavella and Randall (2000) map a uniform computational
coordinate :math:`\xi \in [0, 1]` through a hyperbolic sine,

.. math:: S(\xi) = K + \alpha \sinh\bigl(c_1 + \xi (c_2 - c_1)\bigr), \quad
          c_1 = \operatorname{arsinh}\frac{S_{\min} - K}{\alpha}, \quad
          c_2 = \operatorname{arsinh}\frac{S_{\max} - K}{\alpha},

which clusters nodes around :math:`K`; the smaller :math:`\alpha`, the
tighter the cluster. Because the map is smooth, the three-point
differences of :mod:`financialkit.pde_methods.utils.operators` stay
second-order accurate on it.

Reference: D. Tavella and C. Randall, *Pricing Financial Instruments: The
Finite Difference Method* (Wiley, 2000), Ch. 5.
"""

from __future__ import annotations

import numpy as np

__all__ = ["uniform_grid", "tavella_randall_grid", "default_s_max"]


def default_s_max(S: float, K: float, T: float, sigma: float) -> float:
    r"""Upper edge of the spot grid: :math:`\max(S, K) \max(4, e^{5\sigma\sqrt{T}})`.

    Four times the larger of spot and strike keeps the strike on a node of
    any uniform grid with a multiple of four intervals; the exponential
    term reaches five standard deviations for long or volatile options.

    Parameters
    ----------
    S, K, T, sigma : float

    Returns
    -------
    float

    Examples
    --------
    >>> default_s_max(100.0, 100.0, 1.0, 0.2)
    400.0
    """
    return float(max(S, K) * max(4.0, np.exp(5.0 * sigma * np.sqrt(T))))


def uniform_grid(s_min: float, s_max: float, n: int) -> np.ndarray:
    """Evenly spaced spot grid with `n` intervals.

    Parameters
    ----------
    s_min, s_max : float
    n : int
        Number of intervals.

    Returns
    -------
    ndarray, shape (n + 1,)
    """
    if not s_max > s_min:
        raise ValueError("s_max must exceed s_min")
    return np.linspace(s_min, s_max, int(n) + 1)


def tavella_randall_grid(s_min: float, s_max: float, n: int, center: float, alpha: float) -> np.ndarray:
    """Spot grid clustered around `center` by Tavella and Randall's sinh map.

    Parameters
    ----------
    s_min, s_max : float
        Grid edges.
    n : int
        Number of intervals.
    center : float
        Point to cluster the nodes around, usually the strike.
    alpha : float
        Width of the cluster; uniform as ``alpha -> inf``.

    Returns
    -------
    ndarray, shape (n + 1,)

    Examples
    --------
    >>> g = tavella_randall_grid(0.0, 400.0, 40, center=100.0, alpha=10.0)
    >>> float(g[0]), float(g[-1])
    (0.0, 400.0)
    >>> bool(np.diff(g).min() < 2.0 < 10.0 < np.diff(g).max())
    True
    """
    if not s_min <= center <= s_max:
        raise ValueError("center must lie inside [s_min, s_max]")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    c1, c2 = np.arcsinh((s_min - center) / alpha), np.arcsinh((s_max - center) / alpha)
    grid = center + alpha * np.sinh(c1 + np.linspace(0.0, 1.0, int(n) + 1) * (c2 - c1))
    grid[0], grid[-1] = s_min, s_max
    return grid
