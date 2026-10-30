r"""The Heston PDE by ADI, after in 't Hout and Foulon (2010).

Under Heston's model the value :math:`u(s, v, \tau)` of a European option
solves, in time to maturity :math:`\tau`,

.. math:: u_\tau = \underbrace{\rho\xi v s\, u_{sv}}_{F_0}
          + \underbrace{\tfrac12 v s^2 u_{ss} + (r - q) s\, u_s - \tfrac12 r u}_{F_1}
          + \underbrace{\tfrac12 \xi^2 v\, u_{vv} + \kappa(\theta - v) u_v - \tfrac12 r u}_{F_2}

on :math:`[0, S_{\max}] \times [0, V_{\max}]`. In 't Hout and Foulon
discretise it on non-uniform grids, uniform in :math:`s` around the
strike and hyperbolic-sine outside, and clustered near :math:`v = 0` in
the variance, with second-order central differences, a forward
difference at :math:`v = 0` and a backward (upwind) difference for the
convection where :math:`v > 1`. For a call the boundary conditions are

.. math:: u(0, v, \tau) = 0, \qquad u_s(S_{\max}, v, \tau) = e^{-q\tau}, \qquad
          u(s, V_{\max}, \tau) = s e^{-q\tau},

and at :math:`v = 0` the PDE itself holds, degenerate as it is: the
diffusion and mixed terms vanish and :math:`\kappa\theta u_v` carries the
solution into the domain whether or not Feller's condition holds. The
Neumann condition enters through a ghost node,
:math:`u_{m_1 + 1} = u_{m_1 - 1} + 2 h e^{-q\tau}`. They then compared
the Douglas, Craig-Sneyd, modified Craig-Sneyd and Hundsdorfer-Verwer
splittings on six parameter sets and found the last two second order,
stable and accurate even with the mixed derivative large; put prices
follow from put-call parity, which the discrete solution satisfies
exactly because the forward :math:`s e^{-q\tau} - K e^{-r\tau}` is a
solution of the scheme too.

References: K. J. in 't Hout and S. Foulon, "ADI Finite Difference
Schemes for Option Pricing in the Heston Model with Correlation,"
*Int. J. Numer. Anal. Model.* 7 (2010) 303-320; S. L. Heston,
*Rev. Financ. Stud.* 6 (1993) 327-343.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RectBivariateSpline

from financialkit.options.utils.common import option_sign
from financialkit.pde_methods.core.base import HestonPDEResult
from financialkit.pde_methods.utils.splitting import adi_march, default_adi_theta, difference_matrices

__all__ = ["HestonADI", "heston_spot_grid", "heston_variance_grid"]


def heston_spot_grid(m: int, K: float, s_max: float, T: float = 1.0) -> np.ndarray:
    """In 't Hout and Foulon's spot grid: uniform near the strike, sinh-stretched outside.

    Nodes are uniform on :math:`[S_l, K]` with
    :math:`S_l = \\max(1/2, e^{-T/4}) K` and spaced by a hyperbolic sine of
    width :math:`c = K/5` below :math:`S_l` and above :math:`K`.

    Parameters
    ----------
    m : int
        Number of intervals.
    K : float
        Strike.
    s_max : float
        Upper edge of the grid.
    T : float
        Maturity, which sets the uniform region.

    Returns
    -------
    ndarray, shape (m + 1,)

    Examples
    --------
    >>> g = heston_spot_grid(50, 100.0, 800.0)
    >>> float(g[0]), float(g[-1]), bool(np.diff(g).min() < 5.0 < 50.0 < np.diff(g).max())
    (0.0, 800.0, True)
    """
    if int(m) != m or m < 4:
        raise ValueError(f"m must be an integer >= 4, got {m}")
    if not s_max > K > 0:
        raise ValueError("need s_max > K > 0")
    s_left, c = max(0.5, np.exp(-T / 4.0)) * K, K / 5.0
    xi_int = (K - s_left) / c
    xi = np.linspace(np.arcsinh(-s_left / c), xi_int + np.arcsinh((s_max - K) / c), int(m) + 1)
    grid = np.where(xi < 0.0, s_left + c * np.sinh(xi), np.where(xi <= xi_int, s_left + c * xi, K + c * np.sinh(xi - xi_int)))
    grid[0], grid[-1] = 0.0, s_max
    return grid


def heston_variance_grid(m: int, v_max: float = 5.0) -> np.ndarray:
    """In 't Hout and Foulon's variance grid :math:`v_j = d \\sinh(j \\Delta\\eta)`, :math:`d = V_{\\max}/500`.

    Parameters
    ----------
    m : int
        Number of intervals.
    v_max : float
        Upper edge of the grid.

    Returns
    -------
    ndarray, shape (m + 1,)

    Examples
    --------
    >>> g = heston_variance_grid(25)
    >>> float(g[0]), float(g[-1]), bool(g[1] < 0.01)
    (0.0, 5.0, True)
    """
    if int(m) != m or m < 3:
        raise ValueError(f"m must be an integer >= 3, got {m}")
    if v_max <= 0:
        raise ValueError("v_max must be positive")
    d = v_max / 500.0
    grid = d * np.sinh(np.linspace(0.0, np.arcsinh(v_max / d), int(m) + 1))
    grid[-1] = v_max
    return grid


def _one_sided(v: np.ndarray, upwind: np.ndarray) -> sp.csr_matrix:
    """First-derivative matrix in `v`: forward at ``v[0]``, backward where `upwind`, central elsewhere."""
    D = sp.lil_matrix(difference_matrices(v)[0])
    h1, h2 = v[1] - v[0], v[2] - v[1]
    D[0, :3] = [-(2 * h1 + h2) / (h1 * (h1 + h2)), (h1 + h2) / (h1 * h2), -h1 / (h2 * (h1 + h2))]
    for j in np.flatnonzero(upwind):
        h1, h2 = v[j - 1] - v[j - 2], v[j] - v[j - 1]
        D[j, j - 2 : j + 1] = [h2 / (h1 * (h1 + h2)), -(h1 + h2) / (h1 * h2), (h1 + 2 * h2) / (h2 * (h1 + h2))]
        if j + 1 < v.size:
            D[j, j + 1] = 0.0
    return sp.csr_matrix(D)


class HestonADI:
    """European options under Heston's model by finite differences and ADI.

    Parameters
    ----------
    m1, m2 : int
        Number of spot and variance intervals.
    n_time : int
        Number of time steps.
    scheme : {"modified_craig_sneyd", "craig_sneyd", "hundsdorfer_verwer", "douglas"}
        Splitting scheme.
    adi_theta : float, optional
        Implicitness of the sweeps; defaults to
        :func:`~financialkit.pde_methods.utils.splitting.default_adi_theta`.
    s_max_factor : float
        :math:`S_{\\max}` as a multiple of the strike.
    v_max : float
        Upper edge of the variance grid.

    Examples
    --------
    In 't Hout and Foulon's case A with
    :math:`v_0 = \\theta`; Heston's formula gives 8.8949:

    >>> res = HestonADI(100, 50, 50).price(100.0, 100.0, 1.0, 0.025, 0.04, 1.5, 0.04, 0.3, -0.9)
    >>> round(res.price, 2)
    8.89
    """

    def __init__(
        self,
        m1: int = 50,
        m2: int = 25,
        n_time: int = 20,
        scheme: str = "modified_craig_sneyd",
        adi_theta: float | None = None,
        s_max_factor: float = 8.0,
        v_max: float = 5.0,
    ):
        if int(n_time) != n_time or n_time < 1:
            raise ValueError(f"n_time must be a positive integer, got {n_time}")
        if s_max_factor <= 1:
            raise ValueError("s_max_factor must exceed 1")
        self.m1, self.m2, self.n_time = int(m1), int(m2), int(n_time)
        self.scheme, self.adi_theta = scheme, default_adi_theta(scheme) if adi_theta is None else float(adi_theta)
        self.s_max_factor, self.v_max = float(s_max_factor), float(v_max)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> HestonPDEResult:
        """Value a European call or put.

        Parameters
        ----------
        S, K, T, r : float
            Spot, strike, maturity and risk-free rate.
        v0 : float
            Initial variance.
        kappa, theta, xi, rho : float
            Mean-reversion speed, long-run variance, volatility of variance
            and correlation, as in
            :class:`~financialkit.stochastic_volatility.HestonModel`.
        q : float
            Dividend yield.
        option_type : {"call", "put"}

        Returns
        -------
        HestonPDEResult
        """
        sign = option_sign(option_type)
        if not (S > 0 and K > 0 and T > 0 and kappa > 0 and theta > 0 and xi > 0 and v0 >= 0):
            raise ValueError("need S, K, T, kappa, theta, xi > 0 and v0 >= 0")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        s = heston_spot_grid(self.m1, K, self.s_max_factor * K, T)
        v = heston_variance_grid(self.m2, self.v_max)
        if not (S < s[-1] and v0 < v[-1]):
            raise ValueError("(S, v0) must lie inside the grid")
        Ds, Dss = difference_matrices(s)
        h = s[-1] - s[-2]
        Dss = sp.lil_matrix(Dss)
        Dss[-1, -2:] = [2.0 / h**2, -2.0 / h**2]
        Dv, Dvv = difference_matrices(v)
        Dv_conv = _one_sided(v, v > 1.0)
        S2, V2 = (a.ravel() for a in np.meshgrid(s, v, indexing="ij"))
        Is, Iv = sp.identity(s.size), sp.identity(v.size)
        full = (
            sp.diags(rho * xi * S2 * V2) @ sp.kron(Ds, Dv),
            sp.diags(0.5 * V2 * S2**2) @ sp.kron(Dss, Iv) + sp.diags((r - q) * S2) @ sp.kron(Ds, Iv) - 0.5 * r * sp.identity(S2.size),
            sp.diags(0.5 * xi**2 * V2) @ sp.kron(Is, Dvv) + sp.diags(kappa * (theta - V2)) @ sp.kron(Is, Dv_conv) - 0.5 * r * sp.identity(S2.size),
        )
        # Dirichlet nodes: s = 0 and v = v_max; the s = s_max column is unknown (Neumann)
        known = (S2 == 0.0) | (V2 == v[-1])
        rows = [sp.csr_matrix(L)[~known] for L in full]
        A, B = [L[:, ~known] for L in rows], [L[:, known] for L in rows]
        neumann = np.where(S2 == s[-1], V2 * S2**2 / h + (r - q) * S2, 0.0)[~known]

        def g(tau: float) -> list[np.ndarray]:
            edge = (S2 * np.exp(-q * tau))[known]
            g0, g1, g2 = (b @ edge for b in B)
            return [g0, g1 + neumann * np.exp(-q * tau), g2]

        payoff = np.maximum(S2 - K, 0.0)
        U = adi_march(A, g, payoff[~known], T, self.n_time, self.scheme, self.adi_theta)
        call = (S2 * np.exp(-q * T)) * known
        call[~known] = U
        V = call.reshape(s.size, v.size)
        if sign < 0:
            V = V - s[:, None] * np.exp(-q * T) + K * np.exp(-r * T)
        spline = RectBivariateSpline(s, v, V)
        return HestonPDEResult(
            price=float(spline(S, v0)[0, 0]),
            delta=float(spline(S, v0, dx=1)[0, 0]),
            S=s,
            v=v,
            V=V,
            n_time=self.n_time,
            method=self.scheme,
        )
