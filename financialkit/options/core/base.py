"""Abstract base classes and result containers for financialkit.options.

The closed-form pricers return plain floats or arrays, broadcasting over
their inputs, so a smile or a surface is one call. Everything with more
structure returns a small dataclass that the visualizers plot: the
Greeks, a lattice valuation, an American approximation, an implied
volatility smile, a risk-neutral density and a delta-hedging experiment.
The binomial and trinomial trees share the :class:`LatticePricer` ABC so
that they are interchangeable in convergence studies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Greeks",
    "TreeResult",
    "AmericanApproximationResult",
    "SmileResult",
    "DensityResult",
    "HedgeResult",
    "LatticePricer",
]


@dataclass
class Greeks:
    """Sensitivities of an option price, each broadcast like the inputs.

    Theta is :math:`\\partial V / \\partial t = -\\partial V / \\partial T`,
    per year; vega, rho and epsilon are per unit (not per percentage point)
    of volatility, rate and dividend yield.
    """

    delta: np.ndarray | float
    """Sensitivity to the spot, :math:`\\partial V / \\partial S`."""

    gamma: np.ndarray | float
    """Curvature in the spot, :math:`\\partial^2 V / \\partial S^2`."""

    vega: np.ndarray | float
    """Sensitivity to the volatility, :math:`\\partial V / \\partial \\sigma`."""

    theta: np.ndarray | float
    """Time decay, :math:`\\partial V / \\partial t`, per year."""

    rho: np.ndarray | float
    """Sensitivity to the risk-free rate, :math:`\\partial V / \\partial r`."""

    epsilon: np.ndarray | float
    """Sensitivity to the dividend yield, :math:`\\partial V / \\partial q`."""

    vanna: np.ndarray | float
    """Cross derivative :math:`\\partial^2 V / \\partial S \\partial \\sigma`."""

    volga: np.ndarray | float
    """Second derivative in the volatility, :math:`\\partial^2 V / \\partial \\sigma^2`."""


@dataclass
class TreeResult:
    """Output of a lattice (binomial or trinomial) valuation."""

    price: float
    """Option value at the root of the lattice."""

    delta: float
    """Delta read off the first time step of the lattice."""

    gamma: float
    """Gamma read off the second time step of the lattice."""

    n_steps: int
    """Number of time steps actually used (Leisen-Reimer rounds up to odd)."""

    method: str
    """``"crr"``, ``"rendleman_bartter"``, ``"leisen_reimer"`` or ``"trinomial"``."""

    times: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (n_steps + 1,): The lattice's time grid."""

    exercise_boundary: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (n_steps + 1,): For an American put, the highest node
    price at each time where early exercise is optimal (for a call, the
    lowest); ``nan`` where no node is exercised. Empty for European options."""


@dataclass
class AmericanApproximationResult:
    """Output of the Barone-Adesi-Whaley quadratic approximation."""

    price: float
    """Approximate American option value."""

    european_price: float
    """The European (Black-Scholes-Merton) value it adds the premium to."""

    critical_price: float
    """Spot at which immediate exercise becomes optimal (``inf`` for a call,
    ``0`` for a put, when early exercise is never optimal)."""


@dataclass
class SmileResult:
    """Implied volatilities across strikes at one maturity."""

    strikes: np.ndarray
    """ndarray, shape (n,): Strikes."""

    vols: np.ndarray
    """ndarray, shape (n,): Black-Scholes implied volatilities."""

    forward: float
    """Forward price :math:`S e^{(r - q) T}`."""

    T: float
    """Maturity in years."""

    @property
    def log_moneyness(self) -> np.ndarray:
        """ndarray: :math:`\\ln(K / F)` for each strike."""
        return np.log(self.strikes / self.forward)


@dataclass
class DensityResult:
    """A risk-neutral density recovered from option prices."""

    strikes: np.ndarray
    """ndarray, shape (m,): Strikes at which the density is estimated."""

    density: np.ndarray
    """ndarray, shape (m,): Risk-neutral density of :math:`S_T` there."""


@dataclass
class HedgeResult:
    """Output of a discretely rebalanced delta-hedging simulation."""

    times: np.ndarray
    """ndarray, shape (n_rebalance + 1,): Rebalancing dates."""

    spot: np.ndarray
    """ndarray, shape (n_paths, n_rebalance + 1): Simulated spot paths."""

    pnl: np.ndarray
    """ndarray, shape (n_paths,): Terminal profit of the short, hedged
    option position, in time-``T`` money."""

    option_price: float
    """Premium received for the option at time 0."""


class LatticePricer(ABC):
    """A recombining lattice for European and American vanilla options.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    """

    method: str = ""

    def __init__(self, n_steps: int = 200):
        if int(n_steps) != n_steps or n_steps < 1:
            raise ValueError(f"n_steps must be a positive integer, got {n_steps}")
        self.n_steps = int(n_steps)

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
    ) -> TreeResult:
        """Value a vanilla option on the lattice.

        Parameters
        ----------
        S, K, T, r, sigma : float
            Spot, strike, maturity (years), risk-free rate and volatility.
        q : float
            Continuous dividend yield.
        option_type : {"call", "put"}
        american : bool
            Allow early exercise at every node.

        Returns
        -------
        TreeResult
        """
