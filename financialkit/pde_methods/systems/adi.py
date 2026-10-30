r"""Peaceman-Rachford alternating-direction implicit scheme for two assets.

A two-asset option solves, in :math:`x = \ln S_1`, :math:`y = \ln S_2` and
time to maturity :math:`\tau`,

.. math:: V_\tau = \underbrace{\tfrac12\sigma_1^2 V_{xx} + \nu_1 V_x - \tfrac r2 V}_{A_1 V}
          + \underbrace{\tfrac12\sigma_2^2 V_{yy} + \nu_2 V_y - \tfrac r2 V}_{A_2 V}
          + \underbrace{\rho\sigma_1\sigma_2 V_{xy}}_{A_0 V},
          \qquad \nu_i = r - q_i - \tfrac12\sigma_i^2.

A fully implicit two-dimensional step is a large banded system.
Peaceman and Rachford (1955) split each step into two half steps, each
implicit in one direction only, so that every half step is a set of
independent tridiagonal solves:

.. math:: (I - \tfrac{\Delta\tau}{2} A_1) V^{*} = (I + \tfrac{\Delta\tau}{2} (A_2 + A_0)) V^n, \qquad
          (I - \tfrac{\Delta\tau}{2} A_2) V^{n+1} = (I + \tfrac{\Delta\tau}{2} (A_1 + A_0)) V^{*}.

Without correlation this is the original, unconditionally stable and
second-order scheme; the mixed derivative :math:`A_0` is taken explicitly,
which is the simplest way to keep each half step tridiagonal. Along each
grid line the matrix is the same, so one :func:`scipy.linalg.solve_banded`
call handles all lines at once. Boundary values are the discounted payoff
of the forwards, :math:`e^{-r\tau} f(S_1 e^{(r - q_1)\tau}, S_2 e^{(r - q_2)\tau})`.

Taking :math:`A_0` explicitly costs accuracy once the correlation is
large. Craig and Sneyd (1988) keep the one-direction implicit sweeps but
follow them with a corrector that re-evaluates the mixed term at the new
time and sweeps again; with :math:`\theta = 1/2` the scheme is second
order in time whatever the correlation, where the Douglas scheme it
corrects is first order. :class:`CraigSneydADI` builds the three
operators as sparse matrices and steps them with
:func:`~financialkit.pde_methods.utils.splitting.adi_march`.

References: D. W. Peaceman and H. H. Rachford, *J. SIAM* 3 (1955) 28-41;
I. J. D. Craig and A. D. Sneyd, *Comput. Math. Appl.* 16 (1988) 341-350;
K. J. in 't Hout and S. Foulon, *Int. J. Numer. Anal. Model.* 7 (2010)
303-320; Tavella and Randall, *Pricing Financial Instruments* (Wiley,
2000), Ch. 7.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from financialkit.pde_methods.core.base import ADIResult
from financialkit.pde_methods.utils.operators import solve_tridiagonal
from financialkit.pde_methods.utils.splitting import adi_march, default_adi_theta, difference_matrices

__all__ = ["PeacemanRachfordADI", "CraigSneydADI"]


class PeacemanRachfordADI:
    """Two-asset European option by the Peaceman-Rachford ADI scheme.

    Parameters
    ----------
    n1, n2 : int
        Number of log-spot intervals for each asset (rounded up to even, so
        the spots are nodes).
    n_time : int
        Number of full time steps.
    width : float
        Half-width of each grid in standard deviations :math:`\\sigma_i\\sqrt{T}`.

    Examples
    --------
    Margrabe's exchange option, worth Black-Scholes with spot :math:`S_1`,
    strike :math:`S_2`, rate :math:`q_2` and volatility
    :math:`\\sqrt{\\sigma_1^2 + \\sigma_2^2 - 2\\rho\\sigma_1\\sigma_2} = 0.2646`:

    >>> adi = PeacemanRachfordADI(100, 100, 50)
    >>> res = adi.price(100.0, 100.0, 1.0, 0.05, 0.2, 0.3, 0.5, lambda s1, s2: np.maximum(s1 - s2, 0.0))
    >>> round(res.price, 1)  # Margrabe: 10.5
    10.5
    """

    def __init__(self, n1: int = 100, n2: int = 100, n_time: int = 50, width: float = 5.0):
        for name, n in (("n1", n1), ("n2", n2), ("n_time", n_time)):
            if int(n) != n or n < 2:
                raise ValueError(f"{name} must be an integer >= 2, got {n}")
        self.n1, self.n2 = int(n1) + int(n1) % 2, int(n2) + int(n2) % 2
        self.n_time, self.width = int(n_time), float(width)

    def price(
        self,
        S1: float,
        S2: float,
        T: float,
        r: float,
        sigma1: float,
        sigma2: float,
        rho: float,
        payoff: Callable[[np.ndarray, np.ndarray], np.ndarray],
        q1: float = 0.0,
        q2: float = 0.0,
    ) -> ADIResult:
        """Value a European payoff ``payoff(S1, S2)`` at maturity `T`.

        Parameters
        ----------
        S1, S2 : float
            Spots.
        T, r : float
            Maturity and risk-free rate.
        sigma1, sigma2 : float
            Volatilities.
        rho : float
            Correlation of the two Brownian motions.
        payoff : callable
            Vectorised ``payoff(S1, S2)`` on meshgrid arrays.
        q1, q2 : float
            Dividend yields.

        Returns
        -------
        ADIResult
        """
        if not (S1 > 0 and S2 > 0 and T > 0 and sigma1 > 0 and sigma2 > 0):
            raise ValueError("S1, S2, T, sigma1 and sigma2 must all be positive")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        h1, h2 = self.width * sigma1 * np.sqrt(T), self.width * sigma2 * np.sqrt(T)
        x = np.log(S1) + np.linspace(-h1, h1, self.n1 + 1)
        y = np.log(S2) + np.linspace(-h2, h2, self.n2 + 1)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(np.exp(x), np.exp(y), indexing="ij")
        nu1, nu2 = r - q1 - 0.5 * sigma1**2, r - q2 - 0.5 * sigma2**2
        a1, b1, c1 = 0.5 * sigma1**2 / dx**2 - 0.5 * nu1 / dx, -(sigma1**2) / dx**2 - 0.5 * r, 0.5 * sigma1**2 / dx**2 + 0.5 * nu1 / dx
        a2, b2, c2 = 0.5 * sigma2**2 / dy**2 - 0.5 * nu2 / dy, -(sigma2**2) / dy**2 - 0.5 * r, 0.5 * sigma2**2 / dy**2 + 0.5 * nu2 / dy
        cross = rho * sigma1 * sigma2 / (4.0 * dx * dy)
        dt = T / self.n_time
        h = 0.5 * dt

        def A1(V):
            return a1 * V[:-2, 1:-1] + b1 * V[1:-1, 1:-1] + c1 * V[2:, 1:-1]

        def A2(V):
            return a2 * V[1:-1, :-2] + b2 * V[1:-1, 1:-1] + c2 * V[1:-1, 2:]

        def A0(V):
            return cross * (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2])

        def edges(tau: float) -> np.ndarray:
            return np.exp(-r * tau) * payoff(X * np.exp((r - q1) * tau), Y * np.exp((r - q2) * tau))

        def implicit(a, b, c, m):
            return np.full(m, -h * a), np.full(m, 1.0 - h * b), np.full(m, -h * c)

        lo1, di1, up1 = implicit(a1, b1, c1, self.n1 - 1)
        lo2, di2, up2 = implicit(a2, b2, c2, self.n2 - 1)
        V = np.asarray(payoff(X, Y), dtype=np.float64)
        for n in range(self.n_time):
            star = edges((n + 0.5) * dt)
            rhs = V[1:-1, 1:-1] + h * (A2(V) + A0(V))
            rhs[0] += h * a1 * star[0, 1:-1]
            rhs[-1] += h * c1 * star[-1, 1:-1]
            star[1:-1, 1:-1] = solve_tridiagonal(lo1, di1, up1, rhs)
            new = edges((n + 1) * dt)
            rhs = star[1:-1, 1:-1] + h * (A1(star) + A0(star))
            rhs[:, 0] += h * a2 * new[1:-1, 0]
            rhs[:, -1] += h * c2 * new[1:-1, -1]
            new[1:-1, 1:-1] = solve_tridiagonal(lo2, di2, up2, rhs.T).T
            V = new
        return ADIResult(price=float(V[self.n1 // 2, self.n2 // 2]), S1=np.exp(x), S2=np.exp(y), V=V, n_time=self.n_time)


class CraigSneydADI:
    """Two-asset European option by the Craig-Sneyd ADI scheme.

    The grid, boundary values and operator split are those of
    :class:`PeacemanRachfordADI`; only the time stepping differs.

    Parameters
    ----------
    n1, n2 : int
        Number of log-spot intervals for each asset (rounded up to even, so
        the spots are nodes).
    n_time : int
        Number of time steps.
    width : float
        Half-width of each grid in standard deviations :math:`\\sigma_i\\sqrt{T}`.
    scheme : {"craig_sneyd", "douglas", "modified_craig_sneyd", "hundsdorfer_verwer"}
        Splitting scheme; ``"douglas"`` is Craig-Sneyd without its corrector.
    adi_theta : float, optional
        Implicitness of the sweeps; defaults to
        :func:`~financialkit.pde_methods.utils.splitting.default_adi_theta`.

    Examples
    --------
    Margrabe's exchange option at a correlation of 0.5 (Margrabe: 10.5):

    >>> cs = CraigSneydADI(100, 100, 25)
    >>> res = cs.price(100.0, 100.0, 1.0, 0.05, 0.2, 0.3, 0.5, lambda s1, s2: np.maximum(s1 - s2, 0.0))
    >>> round(res.price, 1), res.method
    (10.5, 'craig_sneyd')
    """

    def __init__(
        self,
        n1: int = 100,
        n2: int = 100,
        n_time: int = 50,
        width: float = 5.0,
        scheme: str = "craig_sneyd",
        adi_theta: float | None = None,
    ):
        for name, n in (("n1", n1), ("n2", n2), ("n_time", n_time)):
            if int(n) != n or n < 2:
                raise ValueError(f"{name} must be an integer >= 2, got {n}")
        self.n1, self.n2 = int(n1) + int(n1) % 2, int(n2) + int(n2) % 2
        self.n_time, self.width = int(n_time), float(width)
        self.scheme, self.adi_theta = scheme, default_adi_theta(scheme) if adi_theta is None else float(adi_theta)

    def price(
        self,
        S1: float,
        S2: float,
        T: float,
        r: float,
        sigma1: float,
        sigma2: float,
        rho: float,
        payoff: Callable[[np.ndarray, np.ndarray], np.ndarray],
        q1: float = 0.0,
        q2: float = 0.0,
    ) -> ADIResult:
        """Value a European payoff ``payoff(S1, S2)`` at maturity `T`.

        Parameters
        ----------
        S1, S2 : float
            Spots.
        T, r : float
            Maturity and risk-free rate.
        sigma1, sigma2 : float
            Volatilities.
        rho : float
            Correlation of the two Brownian motions.
        payoff : callable
            Vectorised ``payoff(S1, S2)`` on meshgrid arrays.
        q1, q2 : float
            Dividend yields.

        Returns
        -------
        ADIResult
        """
        if not (S1 > 0 and S2 > 0 and T > 0 and sigma1 > 0 and sigma2 > 0):
            raise ValueError("S1, S2, T, sigma1 and sigma2 must all be positive")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        h1, h2 = self.width * sigma1 * np.sqrt(T), self.width * sigma2 * np.sqrt(T)
        x = np.log(S1) + np.linspace(-h1, h1, self.n1 + 1)
        y = np.log(S2) + np.linspace(-h2, h2, self.n2 + 1)
        X, Y = np.meshgrid(np.exp(x), np.exp(y), indexing="ij")
        Dx, Dxx = difference_matrices(x)
        Dy, Dyy = difference_matrices(y)
        Ix, Iy = sp.identity(x.size), sp.identity(y.size)
        nu1, nu2 = r - q1 - 0.5 * sigma1**2, r - q2 - 0.5 * sigma2**2
        full = (
            rho * sigma1 * sigma2 * sp.kron(Dx, Dy),
            sp.kron(0.5 * sigma1**2 * Dxx + nu1 * Dx - 0.5 * r * Ix, Iy),
            sp.kron(Ix, 0.5 * sigma2**2 * Dyy + nu2 * Dy - 0.5 * r * Iy),
        )
        inner = np.zeros(X.shape, dtype=bool)
        inner[1:-1, 1:-1] = True
        inner = inner.ravel()
        rows = [sp.csr_matrix(L)[inner] for L in full]
        A, B = [L[:, inner] for L in rows], [L[:, ~inner] for L in rows]

        def edges(tau: float) -> np.ndarray:
            return np.asarray(np.exp(-r * tau) * payoff(X * np.exp((r - q1) * tau), Y * np.exp((r - q2) * tau)), dtype=np.float64)

        def g(tau: float) -> list[np.ndarray]:
            edge = edges(tau).ravel()[~inner]
            return [b @ edge for b in B]

        V0 = np.asarray(payoff(X, Y), dtype=np.float64).ravel()[inner]
        U = adi_march(A, g, V0, T, self.n_time, self.scheme, self.adi_theta)
        V = edges(T)
        V[1:-1, 1:-1] = U.reshape(self.n1 - 1, self.n2 - 1)
        return ADIResult(price=float(V[self.n1 // 2, self.n2 // 2]), S1=np.exp(x), S2=np.exp(y), V=V, n_time=self.n_time, method=self.scheme)
