r"""Finite-difference theta-schemes for the Black-Scholes equation.

In time to maturity :math:`\tau = T - t` the Black-Scholes equation is a
forward diffusion problem,

.. math:: V_\tau = \tfrac12 \sigma^2 S^2 V_{SS} + (r - q) S V_S - r V, \qquad V(S, 0) = \text{payoff}(S),

solved here on a spot grid :math:`[S_{\min}, S_{\max}]` with Dirichlet
boundaries (the discounted intrinsic value, or zero at a knock-out
barrier) by the theta-method of
:mod:`financialkit.pde_methods.utils.stepping`:

* :class:`ExplicitScheme` (:math:`\theta = 0`), the scheme Schwartz (1977)
  used to value warrants: no linear solve, but stable only for
  :math:`\Delta t \lesssim 1/(\sigma^2 M^2)`.
* :class:`ImplicitScheme` (:math:`\theta = 1`), Laasonen's (1949) fully
  implicit scheme: unconditionally stable and monotone, first order in
  time.
* :class:`CrankNicolson` (:math:`\theta = 1/2`, 1947): unconditionally
  stable and second order in time, with Rannacher's (1984) implicit
  start to damp the oscillations a payoff kink excites.

American options solve a linear complementarity problem at each implicit
step by PSOR, the Brennan-Schwartz algorithm or the penalty method, and
:meth:`ThetaScheme.price_barrier` aligns the grid with a knock-out
barrier as Zvan, Forsyth and Vetzal (2000) recommend. Grids are uniform
or clustered at the strike by Tavella and Randall's sinh map.

References: E. S. Schwartz, *J. Financial Economics* 4 (1977) 79-93;
P. Laasonen, *Acta Math.* 81 (1949) 309-317; J. Crank and P. Nicolson,
*Proc. Camb. Phil. Soc.* 43 (1947) 50-67; R. Rannacher, *Numer. Math.* 43
(1984) 309-327; R. Zvan, P. A. Forsyth and K. R. Vetzal, *J. Economic
Dynamics and Control* 24 (2000) 1563-1590; Hull, *Options, Futures, and
Other Derivatives*, 10th ed., Ch. 21.
"""

from __future__ import annotations

import numpy as np

from financialkit.options.utils.common import option_sign, payoff
from financialkit.pde_methods.core.base import FDResult, FiniteDifferencePricer, LCPSolution
from financialkit.pde_methods.systems.american import brennan_schwartz, penalty_solve, psor
from financialkit.pde_methods.systems.grids import default_s_max, tavella_randall_grid, uniform_grid
from financialkit.pde_methods.systems.stability import explicit_stability
from financialkit.pde_methods.utils.operators import black_scholes_operator, discounted_intrinsic, interpolate_at
from financialkit.pde_methods.utils.stepping import LCPSolver, theta_march

__all__ = ["ThetaScheme", "ExplicitScheme", "ImplicitScheme", "CrankNicolson"]

_METHOD_NAMES = {0.0: "explicit", 0.5: "crank_nicolson", 1.0: "implicit"}
_LCP_METHODS = ("psor", "brennan_schwartz", "penalty")
_BARRIER_TYPES = ("down-and-out", "up-and-out")


def _check_inputs(S: float, K: float, T: float, sigma: float) -> None:
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        raise ValueError("S, K, T and sigma must all be positive")


class ThetaScheme(FiniteDifferencePricer):
    """The theta-method for European, American and knock-out vanilla options.

    Parameters
    ----------
    n_space : int
        Number of spot intervals :math:`M`.
    n_time : int
        Number of time steps :math:`N`.
    theta : float
        Implicitness in ``[0, 1]``.
    rannacher_steps : int
        Initial steps replaced by two implicit half steps each.
    grid : {"uniform", "tavella_randall"}
        Spot grid; ``"tavella_randall"`` clusters the nodes at the strike.
    concentration : float
        Width of the Tavella-Randall cluster as a fraction of the strike.
    s_max : float, optional
        Upper edge of the grid (default :func:`~financialkit.pde_methods.systems.grids.default_s_max`).
    american_method : {"psor", "brennan_schwartz", "penalty"}
        Linear complementarity solver for American implicit steps.
    solver : {"banded", "thomas"}
        Tridiagonal solver for European implicit steps.
    omega : float
        PSOR over-relaxation parameter.

    Examples
    --------
    >>> res = ThetaScheme(400, 200, theta=0.5).price(100.0, 100.0, 1.0, 0.05, 0.2)
    >>> round(res.price, 2)  # Black-Scholes: 10.45
    10.45
    """

    def __init__(
        self,
        n_space: int = 200,
        n_time: int = 200,
        theta: float = 0.5,
        rannacher_steps: int = 0,
        grid: str = "uniform",
        concentration: float = 0.1,
        s_max: float | None = None,
        american_method: str = "psor",
        solver: str = "banded",
        omega: float = 1.2,
    ):
        super().__init__(n_space, n_time)
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {theta}")
        if grid not in ("uniform", "tavella_randall"):
            raise ValueError(f'grid must be "uniform" or "tavella_randall", got {grid!r}')
        if american_method not in _LCP_METHODS:
            raise ValueError(f"american_method must be one of {_LCP_METHODS}, got {american_method!r}")
        self.theta, self.rannacher_steps = float(theta), int(rannacher_steps)
        self.grid, self.concentration, self.s_max = grid, float(concentration), s_max
        self.american_method, self.solver, self.omega = american_method, solver, float(omega)
        self.method = _METHOD_NAMES.get(self.theta, "theta")

    def spot_grid(self, s_min: float, s_max: float, K: float) -> np.ndarray:
        """The spot grid this scheme uses on ``[s_min, s_max]`` for strike `K`.

        Parameters
        ----------
        s_min, s_max, K : float

        Returns
        -------
        ndarray, shape (n_space + 1,)
        """
        if self.grid == "uniform":
            return uniform_grid(s_min, s_max, self.n_space)
        center = min(max(K, s_min), s_max)
        return tavella_randall_grid(s_min, s_max, self.n_space, center, self.concentration * K)

    def _lcp(self, sign: int) -> LCPSolver:
        if self.american_method == "psor":
            return lambda lo, di, up, rhs, g, x0: psor(lo, di, up, rhs, g, x0, omega=self.omega)
        if self.american_method == "penalty":
            return lambda lo, di, up, rhs, g, x0: penalty_solve(lo, di, up, rhs, g, x0)
        side = "low" if sign < 0 else "high"

        def bs(lo, di, up, rhs, g, x0) -> LCPSolution:
            return brennan_schwartz(lo, di, up, rhs, g, exercise=side)

        return bs

    def _solve(self, grid, V0, S, K, T, r, sigma, q, option_type, boundary, american) -> FDResult:
        sign = option_sign(option_type)
        lower, diag, upper = black_scholes_operator(grid, sigma, r, q)
        obstacle = V0.copy() if american else None
        surface, iterations = theta_march(
            V0,
            lower,
            diag,
            upper,
            T,
            self.n_time,
            boundary,
            theta=self.theta,
            rannacher_steps=self.rannacher_steps,
            obstacle=obstacle,
            lcp=self._lcp(sign) if american else None,
            solver=self.solver,
        )
        surface = surface[::-1].copy()
        price, delta, gamma = interpolate_at(grid, surface[0], S)
        boundary_path = np.empty(0)
        if american:
            exercised = (surface - V0 <= 1e-7 * K) & (V0 > 0)
            boundary_path = np.full(self.n_time + 1, np.nan)
            for k, ex in enumerate(exercised):
                if ex.any():
                    boundary_path[k] = grid[ex].max() if sign < 0 else grid[ex].min()
        stability = explicit_stability(grid, T / self.n_time, sigma, r, q) if self.theta == 0.0 else None
        return FDResult(
            price=price,
            delta=delta,
            gamma=gamma,
            S=grid,
            V=surface[0],
            method=self.method,
            times=np.linspace(0.0, T, self.n_time + 1),
            surface=surface,
            exercise_boundary=boundary_path,
            lcp_iterations=iterations,
            stability=stability,
        )

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        american: bool = False,
    ) -> FDResult:
        """Value a vanilla option on a grid from 0 to ``s_max``.

        Parameters
        ----------
        S, K, T, r, sigma : float
            Spot, strike, maturity (years), risk-free rate and volatility.
        q : float
            Continuous dividend yield.
        option_type : {"call", "put"}
        american : bool
            Allow early exercise at every time step.

        Returns
        -------
        FDResult
        """
        _check_inputs(S, K, T, sigma)
        option_sign(option_type)
        s_max = float(self.s_max) if self.s_max is not None else default_s_max(S, K, T, sigma)
        if S >= s_max:
            raise ValueError(f"spot {S} must lie below s_max = {s_max}")
        grid = self.spot_grid(0.0, s_max, K)
        V0 = np.asarray(payoff(grid, K, option_type))
        edges = grid[[0, -1]]

        def boundary(tau: float) -> tuple[float, float]:
            v = discounted_intrinsic(edges, K, tau, r, q, option_type)
            if american:
                v = np.maximum(v, V0[[0, -1]])
            return float(v[0]), float(v[1])

        return self._solve(grid, V0, S, K, T, r, sigma, q, option_type, boundary, american)

    def price_barrier(
        self,
        S: float,
        K: float,
        H: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        barrier_type: str = "down-and-out",
    ) -> FDResult:
        """Value a continuously monitored knock-out option with a node on the barrier.

        The grid ends exactly at the barrier `H`, where the value is set to
        zero (no rebate), so the barrier condition is imposed without
        interpolation error.

        Parameters
        ----------
        S, K, H, T, r, sigma : float
            Spot, strike, barrier, maturity, rate and volatility.
        q : float
            Continuous dividend yield.
        option_type : {"call", "put"}
        barrier_type : {"down-and-out", "up-and-out"}

        Returns
        -------
        FDResult
        """
        _check_inputs(S, K, T, sigma)
        option_sign(option_type)
        if barrier_type not in _BARRIER_TYPES:
            raise ValueError(f"barrier_type must be one of {_BARRIER_TYPES}, got {barrier_type!r}")
        down = barrier_type == "down-and-out"
        if (down and not H < S) or (not down and not S < H):
            raise ValueError(f"the spot must lie on the live side of the barrier for a {barrier_type} option")
        if down:
            s_max = float(self.s_max) if self.s_max is not None else default_s_max(S, K, T, sigma)
            grid = self.spot_grid(H, max(s_max, 2.0 * H), K)
        else:
            grid = self.spot_grid(0.0, H, K)
        V0 = np.asarray(payoff(grid, K, option_type))
        V0[0 if down else -1] = 0.0
        far = grid[-1] if down else grid[0]

        def boundary(tau: float) -> tuple[float, float]:
            v = float(discounted_intrinsic(far, K, tau, r, q, option_type))
            return (0.0, v) if down else (v, 0.0)

        return self._solve(grid, V0, S, K, T, r, sigma, q, option_type, boundary, False)


class ExplicitScheme(ThetaScheme):
    """Schwartz's explicit finite-difference scheme (``theta = 0``).

    Each step is a matrix-vector product, and American exercise is a
    simple ``max`` with the payoff. Stable only while
    :math:`\\Delta t \\lesssim 1/(\\sigma^2 (M-1)^2)` on a uniform grid; the
    result carries the check in ``FDResult.stability``.

    Parameters
    ----------
    n_space, n_time : int
        Spot intervals and time steps.
    grid, concentration, s_max
        As for :class:`ThetaScheme`.

    Examples
    --------
    Hull, 10th ed., Example 15.6 (S = 42, K = 40, r = 10%, sigma = 20%, six months):

    >>> res = ExplicitScheme(320, 2500, s_max=160.0).price(42.0, 40.0, 0.5, 0.10, 0.20)
    >>> round(res.price, 2), res.stability.stable
    (4.76, True)
    """

    def __init__(self, n_space: int = 100, n_time: int = 1000, grid: str = "uniform", concentration: float = 0.1, s_max: float | None = None):
        super().__init__(n_space, n_time, theta=0.0, grid=grid, concentration=concentration, s_max=s_max)


class ImplicitScheme(ThetaScheme):
    """Laasonen's fully implicit scheme (``theta = 1``).

    Unconditionally stable and free of oscillations, but only first-order
    accurate in time.

    Parameters
    ----------
    n_space, n_time : int
        Spot intervals and time steps.
    **kwargs
        Any other :class:`ThetaScheme` option except ``theta``.

    Examples
    --------
    >>> res = ImplicitScheme(400, 800).price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="put")
    >>> round(res.price, 2)  # Black-Scholes: 5.57
    5.57
    """

    def __init__(self, n_space: int = 200, n_time: int = 200, **kwargs):
        super().__init__(n_space, n_time, theta=1.0, **kwargs)


class CrankNicolson(ThetaScheme):
    """The Crank-Nicolson scheme (``theta = 1/2``), optionally with Rannacher's start.

    Second-order accurate in time and unconditionally stable, but its
    amplification factor tends to -1 for the stiffest modes, so a payoff
    kink leaves undamped oscillations in delta and gamma when the time
    step is large; ``rannacher_steps=2`` removes them.

    Parameters
    ----------
    n_space, n_time : int
        Spot intervals and time steps.
    rannacher_steps : int
        Initial steps replaced by two implicit half steps each.
    **kwargs
        Any other :class:`ThetaScheme` option except ``theta``.

    Examples
    --------
    >>> res = CrankNicolson(400, 100, rannacher_steps=2).price(100.0, 100.0, 1.0, 0.05, 0.2)
    >>> round(res.price, 2)
    10.45
    """

    def __init__(self, n_space: int = 200, n_time: int = 200, rannacher_steps: int = 0, **kwargs):
        super().__init__(n_space, n_time, theta=0.5, rannacher_steps=rannacher_steps, **kwargs)
