r"""Stability of the explicit and theta-schemes.

Courant, Friedrichs and Lewy (1928) showed that an explicit difference
scheme can only converge if its time step respects the speed at which
information travels on the grid. For the diffusion part of the
Black-Scholes equation that bound is a *diffusion number*: on a uniform
log-spot grid the explicit scheme needs
:math:`\sigma^2 \Delta t / \Delta x^2 \le 1`, and on a uniform spot grid
:math:`\Delta t \lesssim 1 / (\sigma^2 (M - 1)^2)`, so halving the spot step
quarters the time step.

Von Neumann's analysis (published by Charney, Fjortoft and von Neumann
in 1950, and by Crank and Nicolson in 1947) makes this precise by
inserting a Fourier mode :math:`V_j^n = g^n e^{i j \xi}` into the
constant-coefficient scheme. For the theta-scheme in :math:`x = \ln S`,

.. math:: g(\xi) = \frac{1 + (1 - \theta) z(\xi)}{1 - \theta z(\xi)}, \quad
          z(\xi) = \Delta t \Bigl[\frac{\sigma^2}{\Delta x^2}(\cos\xi - 1)
          + i \frac{\nu}{\Delta x} \sin\xi - r\Bigr], \quad
          \nu = r - q - \tfrac12\sigma^2,

and the scheme is stable when :math:`|g(\xi)| \le 1` for every
:math:`\xi`: always for :math:`\theta \ge 1/2`, only for small steps when
:math:`\theta = 0`.

References: R. Courant, K. Friedrichs and H. Lewy, *Math. Annalen* 100
(1928) 32-74; J. G. Charney, R. Fjortoft and J. von Neumann, *Tellus* 2
(1950) 237-254; J. Crank and P. Nicolson, *Proc. Camb. Phil. Soc.* 43
(1947) 50-67; R. D. Richtmyer and K. W. Morton, *Difference Methods for
Initial-Value Problems*, 2nd ed. (Wiley, 1967), Ch. 4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals

from financialkit.pde_methods.core.base import StabilityResult, VonNeumannResult
from financialkit.pde_methods.utils.operators import black_scholes_operator, tridiagonal_matrix

__all__ = ["explicit_stability", "diffusion_number", "von_neumann_amplification"]


def explicit_stability(S: ArrayLike, dt: float, sigma: float, r: float, q: float = 0.0) -> StabilityResult:
    r"""Check an explicit step against the largest stable step on a spot grid.

    The explicit step multiplies the interior values by :math:`I + \Delta t A`,
    with :math:`A` the discrete Black-Scholes operator, so errors stay
    bounded only while :math:`|1 + \Delta t \lambda| \le 1` for every
    eigenvalue :math:`\lambda` of :math:`A`. The limit returned is the
    largest such step, :math:`\min_\lambda -2\operatorname{Re}\lambda / |\lambda|^2`;
    on a uniform grid from 0 it is close to :math:`1/(\sigma^2 (M - 1)^2)`,
    the CFL-type bound that ties the time step to the square of the spot step.

    Parameters
    ----------
    S : array_like, shape (M + 1,)
        Spot grid.
    dt : float
        Time step.
    sigma, r, q : float

    Returns
    -------
    StabilityResult

    Examples
    --------
    >>> res = explicit_stability(np.linspace(0.0, 400.0, 101), 0.002, 0.2, 0.05)
    >>> round(res.limit, 5), res.stable
    (0.00279, True)
    """
    lam = eigvals(tridiagonal_matrix(*black_scholes_operator(S, sigma, r, q)).toarray())
    limit = float(np.min(-2.0 * lam.real / np.abs(lam) ** 2)) if np.all(lam.real < 0) else 0.0
    return StabilityResult(number=float(dt), limit=limit, stable=bool(dt <= limit), scheme="explicit", quantity="dt")


def diffusion_number(sigma: float, dt: float, dx: float) -> StabilityResult:
    r"""The diffusion number :math:`\sigma^2 \Delta t / \Delta x^2` of an explicit log-spot scheme.

    The explicit scheme in :math:`x = \ln S` (equivalently, the trinomial
    tree it defines) is stable, with a non-negative middle probability,
    when this number is at most 1.

    Parameters
    ----------
    sigma, dt, dx : float

    Returns
    -------
    StabilityResult

    Examples
    --------
    >>> diffusion_number(0.2, 0.01, 0.02).number
    1.0
    """
    lam = float(sigma**2 * dt / dx**2)
    return StabilityResult(number=round(lam, 12), limit=1.0, stable=bool(lam <= 1.0 + 1e-12), scheme="explicit", quantity="diffusion")


def von_neumann_amplification(sigma: float, r: float, dt: float, dx: float, theta: float = 0.5, q: float = 0.0, n_modes: int = 201) -> VonNeumannResult:
    """Amplification factor of the log-spot theta-scheme over the Fourier modes.

    Parameters
    ----------
    sigma, r : float
        Volatility and rate.
    dt, dx : float
        Time step and log-spot step.
    theta : float
        Implicitness: 0 explicit, 1/2 Crank-Nicolson, 1 implicit.
    q : float
        Dividend yield.
    n_modes : int
        Number of phase angles in :math:`[0, \\pi]`.

    Returns
    -------
    VonNeumannResult

    Examples
    --------
    Above a diffusion number of 1 the explicit scheme amplifies the
    highest mode; Crank-Nicolson does not:

    >>> von_neumann_amplification(0.2, 0.0, 0.012, 0.02, theta=0.0).stable
    False
    >>> von_neumann_amplification(0.2, 0.0, 0.012, 0.02, theta=0.5).stable
    True
    """
    if not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta must be in [0, 1], got {theta}")
    xi = np.linspace(0.0, np.pi, int(n_modes))
    nu = r - q - 0.5 * sigma**2
    z = dt * (sigma**2 / dx**2 * (np.cos(xi) - 1.0) + 1j * nu / dx * np.sin(xi) - r)
    return VonNeumannResult(xi=xi, g=(1.0 + (1.0 - theta) * z) / (1.0 - theta * z), theta=float(theta))
