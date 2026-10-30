"""Abstract base classes and result containers for financialkit.pde_methods.

Every one-factor finite-difference pricer works in time to maturity
:math:`\\tau = T - t` on a spot grid :math:`S_0 < \\dots < S_M`, steps the
Black-Scholes operator backward from the payoff and returns an
:class:`FDResult`: the price, delta and gamma at the spot, the whole
value surface and, for American options, the early-exercise boundary.
The theta-schemes share the :class:`FiniteDifferencePricer` ABC so that
they are interchangeable in convergence studies, just as the options
lattices share :class:`~financialkit.options.core.base.LatticePricer`.

The remaining dataclasses hold a two-factor ADI solve, a Heston PDE
solve, one solve of a
linear complementarity problem, a stability check, a von Neumann
amplification factor and a Richardson extrapolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from financialkit._deprecation import deprecated_field

__all__ = [
    "FDResult",
    "HeatEquationResult",
    "TrinomialProbabilities",
    "ADIResult",
    "HestonPDEResult",
    "LCPSolution",
    "StabilityResult",
    "VonNeumannResult",
    "RichardsonResult",
    "FiniteDifferencePricer",
]


@dataclass
class StabilityResult:
    """Outcome of a stability check for one time step."""

    number: float
    """The quantity checked: a time step, or a dimensionless diffusion number."""

    limit: float
    """Largest stable value of `number` (``inf`` if the scheme is unconditionally stable)."""

    stable: bool
    """Whether ``number <= limit``."""

    scheme: str = ""
    """Scheme the limit applies to."""

    quantity: str = ""
    """``"dt"`` or ``"diffusion"``."""


@dataclass
class FDResult:
    """Output of a one-factor finite-difference valuation."""

    price: float
    """Option value at the spot, interpolated from the grid by a cubic spline."""

    delta: float
    """:math:`\\partial V / \\partial S` at the spot, from the same spline."""

    gamma: float
    """:math:`\\partial^2 V / \\partial S^2` at the spot, from the same spline."""

    S: np.ndarray
    """ndarray, shape (M + 1,): Spot grid, boundaries included."""

    V: np.ndarray
    """ndarray, shape (M + 1,): Option values on the grid today (``t = 0``)."""

    method: str
    """Scheme used, e.g. ``"explicit"``, ``"implicit"``, ``"crank_nicolson"``."""

    times: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (N + 1,): Calendar times ``t`` from 0 to ``T``."""

    surface: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    """ndarray, shape (N + 1, M + 1): ``surface[k]`` holds the values at ``times[k]``."""

    exercise_boundary: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (N + 1,): For an American put, the highest grid spot at
    each time where the option is exercised (for a call, the lowest); ``nan``
    where no node is exercised. Empty for European options."""

    lcp_iterations: int = 0
    """Total iterations of the linear complementarity solver (American only)."""

    stability: StabilityResult | None = None
    """Stability check of the time step, for explicit schemes."""


@dataclass
class HeatEquationResult(FDResult):
    """An :class:`FDResult` that also keeps the solve in heat-equation variables."""

    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (M + 1,): Log-moneyness grid :math:`x = \\ln(S/K)`."""

    tau: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (N + 1,): Scaled times to maturity :math:`\\tau = \\sigma^2 (T - t) / 2`, ascending."""

    u: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    """ndarray, shape (N + 1, M + 1): Heat-equation solution, ``u[n]`` at ``tau[n]``."""

    alpha: float = 0.0
    """Exponent :math:`\\alpha` in :math:`V = K e^{\\alpha x + \\beta\\tau} u`."""

    beta: float = 0.0
    """Exponent :math:`\\beta` in :math:`V = K e^{\\alpha x + \\beta\\tau} u`."""


@dataclass
class TrinomialProbabilities:
    """Weights of the explicit log-spot scheme read as a trinomial tree."""

    p_down: float
    """Weight on the node one log-step down."""

    p_mid: float
    """Weight on the same node."""

    p_up: float
    """Weight on the node one log-step up."""

    discount: float
    """One-step discount factor :math:`1 / (1 + r \\Delta t)`."""

    @property
    def total(self) -> float:
        """float: Sum of the three weights (1 by construction)."""
        return self.p_down + self.p_mid + self.p_up

    @property
    def is_probability(self) -> bool:
        """bool: Whether all three weights are non-negative."""
        return min(self.p_down, self.p_mid, self.p_up) >= 0.0


@dataclass
class ADIResult:
    """Output of a two-factor alternating-direction implicit solve."""

    price: float
    """Option value at the spots ``(S1, S2)``."""

    S1: np.ndarray
    """ndarray, shape (n1 + 1,): Grid of the first asset."""

    S2: np.ndarray
    """ndarray, shape (n2 + 1,): Grid of the second asset."""

    V: np.ndarray
    """ndarray, shape (n1 + 1, n2 + 1): Values today, ``V[i, j]`` at ``(S1[i], S2[j])``."""

    n_time: int
    """Number of full time steps."""

    method: str = "peaceman_rachford"
    """Scheme used."""


@dataclass
class HestonPDEResult:
    """Output of a finite-difference solve of the Heston PDE."""

    price: float
    """Option value at the spot and initial variance ``(S, v0)``."""

    delta: float
    """:math:`\\partial V / \\partial S` at ``(S, v0)``, from a bicubic spline."""

    S: np.ndarray
    """ndarray, shape (m1 + 1,): Spot grid, from 0 to ``S_max``."""

    v: np.ndarray
    """ndarray, shape (m2 + 1,): Variance grid, from 0 to ``v_max``."""

    V: np.ndarray
    """ndarray, shape (m1 + 1, m2 + 1): Values today, ``V[i, j]`` at ``(S[i], v[j])``."""

    n_time: int
    """Number of time steps."""

    method: str
    """Splitting scheme used."""


@dataclass
class LCPSolution:
    """Solution of one linear complementarity problem
    :math:`Ax \\ge b,\\ x \\ge g,\\ (Ax - b)^\\top (x - g) = 0`."""

    x: np.ndarray
    """ndarray, shape (m,): The solution."""

    n_iter: int
    """Iterations used (1 for a direct method)."""
    iterations = deprecated_field("n_iter")

    method: str
    """``"psor"``, ``"brennan_schwartz"`` or ``"penalty"``."""


@dataclass
class VonNeumannResult:
    """Amplification factor :math:`g(\\xi)` of a scheme over the Fourier modes."""

    xi: np.ndarray
    """ndarray, shape (n,): Phase angles :math:`\\xi = k \\Delta x` in :math:`[0, \\pi]`."""

    g: np.ndarray
    """ndarray, shape (n,): Complex amplification factor of each mode per step."""

    theta: float
    """Implicitness of the theta-scheme (0 explicit, 1/2 Crank-Nicolson, 1 implicit)."""

    @property
    def max_modulus(self) -> float:
        """float: :math:`\\max_\\xi |g(\\xi)|`."""
        return float(np.max(np.abs(self.g)))

    @property
    def stable(self) -> bool:
        """bool: Whether no mode grows, :math:`\\max |g| \\le 1` (to rounding)."""
        return self.max_modulus <= 1.0 + 1e-12


@dataclass
class RichardsonResult:
    """A sequence of approximations on refined grids and its Richardson extrapolation."""

    values: np.ndarray
    """ndarray, shape (k,): Approximations, coarsest first."""

    extrapolated: float
    """Extrapolated value from the two finest approximations."""

    order: float
    """Error order :math:`p` used in the extrapolation."""

    ratio: float
    """Refinement ratio between consecutive grids."""

    observed_order: float = float("nan")
    """Order estimated from the three finest values (``nan`` with fewer than three)."""


class FiniteDifferencePricer(ABC):
    """A finite-difference solver for European and American vanilla options.

    Parameters
    ----------
    n_space : int
        Number of spot intervals :math:`M`.
    n_time : int
        Number of time steps :math:`N`.
    """

    method: str = ""

    def __init__(self, n_space: int = 200, n_time: int = 200):
        for name, n in (("n_space", n_space), ("n_time", n_time)):
            if int(n) != n or n < 2:
                raise ValueError(f"{name} must be an integer >= 2, got {n}")
        self.n_space, self.n_time = int(n_space), int(n_time)

    @abstractmethod
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
        """Value a vanilla option on the grid.

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
