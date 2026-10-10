"""Abstract base classes and result containers for financialkit.interest_rate_models.

Every model here is a :class:`ShortRateModel`: it prices a zero-coupon
bond :math:`P(t, T)` in closed form given the state at time :math:`t`, and
simulates short-rate paths under the risk-neutral measure through
:mod:`financialkit.integrators`. A model's initial term structure is a
:class:`~financialkit.fixed_income.core.base.YieldCurve`
(:meth:`ShortRateModel.curve`), so swap rates, forwards and par yields
come from :mod:`financialkit.fixed_income` unchanged.

Models whose short rate is Gaussian (Merton, Vasicek, Ho-Lee, Hull-White,
G2++) also implement :class:`GaussianShortRateModel`: the log-price of a
bond at a future date is normal, so a zero-coupon bond option is a
Black-76 option on the bond's forward price, and only the total
volatility :meth:`~GaussianShortRateModel.bond_option_volatility` is
model specific.

Results with structure are small dataclasses, never bare tuples, so the
visualizers have a stable interface: Monte Carlo bond prices, a binomial
short-rate lattice, a cap or floor strip, Jamshidian's decomposition and
an HJM forward-curve simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid

from financialkit.fixed_income.core.base import YieldCurve
from financialkit.integrators import RNGLike

__all__ = [
    "ShortRateModel",
    "GaussianShortRateModel",
    "ModelCurve",
    "BondMonteCarloResult",
    "RateTree",
    "CapFloorResult",
    "JamshidianResult",
    "HJMResult",
]


class ShortRateModel(ABC):
    """A risk-neutral model of the instantaneous short rate :math:`r(t)`.

    Subclasses set :attr:`r0` and :attr:`n_factors` and implement
    :meth:`zero_coupon_bond` and :meth:`simulate`.
    """

    r0: float = 0.0
    """Short rate at time 0."""

    n_factors: int = 1
    """Number of Brownian drivers; one-factor models have the short rate as their state."""

    @abstractmethod
    def zero_coupon_bond(self, T, t: float = 0.0, state=None):
        """Price :math:`P(t, T)` of a unit zero-coupon bond.

        Parameters
        ----------
        T : array_like
            Bond maturities, ``T >= t``.
        t : float
            Valuation date.
        state : float or array_like, optional
            Model state at ``t``: the short rate for a one-factor model
            (default :attr:`r0`, only meaningful at ``t = 0``), the factor
            vector otherwise. Broadcasts against ``T`` along leading axes.

        Returns
        -------
        float or ndarray
        """

    @abstractmethod
    def simulate(self, T: float, n_steps: int, n_paths: int, rng: RNGLike = None) -> tuple[np.ndarray, np.ndarray]:
        """Risk-neutral short-rate paths on a uniform grid over ``[0, T]``.

        Returns
        -------
        times : ndarray, shape (n_steps + 1,)
        rates : ndarray, shape (n_paths, n_steps + 1)
        """

    def yields(self, T, t: float = 0.0, state=None):
        """Continuously compounded zero yields :math:`-\\ln P(t, T) / (T - t)`."""
        T = np.asarray(T, dtype=np.float64)
        tau = T - t
        with np.errstate(divide="ignore", invalid="ignore"):
            y = -np.log(self.zero_coupon_bond(T, t, state)) / tau
        if state is None and t == 0.0:
            y = np.where(tau > 0, y, self.r0)
        return y

    def curve(self) -> YieldCurve:
        """The model's initial term structure as a :class:`~financialkit.fixed_income.core.base.YieldCurve`."""
        return ModelCurve(self)


class GaussianShortRateModel(ShortRateModel):
    """A short-rate model in which bond log-prices are Gaussian.

    Then :math:`P(T, S) / P(0, T, S)` is lognormal under the
    :math:`T`-forward measure, and a European option on a zero-coupon bond
    is priced by Black-76 with total volatility :math:`\\sigma_P` given by
    :meth:`bond_option_volatility` (Jamshidian 1989).
    """

    @abstractmethod
    def bond_option_volatility(self, T, S):
        """Total volatility :math:`\\sigma_P` of :math:`\\ln P(T, S)` seen from time 0.

        Parameters
        ----------
        T : array_like
            Option expiry.
        S : array_like
            Maturity of the underlying bond, ``S > T``.

        Returns
        -------
        float or ndarray
            Standard deviation of :math:`\\ln P(T, S)` (not annualised).
        """


class ModelCurve(YieldCurve):
    """A model's time-0 zero curve, :math:`r(t) = -\\ln P(0, t) / t`, with :math:`r(0) = r_0`."""

    def __init__(self, model: ShortRateModel):
        self.model = model

    def zero_rate(self, t):
        t = np.asarray(t, dtype=np.float64)
        safe = np.where(t > 0, t, 1.0)
        return np.where(t > 0, -np.log(self.model.zero_coupon_bond(safe)) / safe, self.model.r0)

    def discount(self, t):
        t = np.asarray(t, dtype=np.float64)
        return np.where(t > 0, self.model.zero_coupon_bond(np.where(t > 0, t, 1.0)), 1.0)

    def __repr__(self) -> str:
        return f"ModelCurve({self.model!r})"


@dataclass
class BondMonteCarloResult:
    """Zero-coupon bond prices :math:`E[e^{-\\int_0^T r\\,dt}]` estimated by simulation (Feynman-Kac)."""

    maturities: np.ndarray
    """ndarray, shape (m,): Bond maturities."""

    prices: np.ndarray
    """ndarray, shape (m,): Monte Carlo prices."""

    std_errors: np.ndarray
    """ndarray, shape (m,): Standard errors of the prices."""

    exact: np.ndarray
    """ndarray, shape (m,): The model's closed-form prices, for comparison."""

    n_paths: int = 0
    """int: Number of simulated paths."""

    @property
    def errors(self) -> np.ndarray:
        """ndarray: Monte Carlo minus closed-form prices."""
        return self.prices - self.exact

    @property
    def z_scores(self) -> np.ndarray:
        """ndarray: Errors in units of standard error."""
        return self.errors / self.std_errors


@dataclass
class RateTree:
    """A recombining binomial short-rate lattice with probability 1/2 on each branch.

    ``rates[i, j]`` is the rate over ``[t_i, t_{i+1}]`` at node ``j``
    (``j = 0`` lowest, ``j <= i``); entries with ``j > i`` are ``nan``.
    Rates are continuously compounded over one step of length :attr:`dt`.
    """

    times: np.ndarray
    """ndarray, shape (n + 1,): Node dates ``0, dt, ..., n dt``."""

    rates: np.ndarray
    """ndarray, shape (n, n): Short rate at each node of steps ``0 .. n-1``."""

    arrow_debreu: np.ndarray
    """ndarray, shape (n + 1, n + 1): Price at time 0 of 1 paid at node ``(i, j)``."""

    dt: float
    """float: Step length in years."""

    sigma: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (n,): Short-rate (log) volatility used at each step."""

    @property
    def n_steps(self) -> int:
        """int: Number of steps."""
        return self.rates.shape[0]

    def _step_index(self, T: float) -> int:
        k = T / self.dt
        i = int(round(k))
        if abs(k - i) > 1e-9 or not 0 <= i <= self.n_steps:
            raise ValueError(f"T = {T} is not a node date of the tree")
        return i

    def _rollback(self, values: np.ndarray, i: int) -> np.ndarray:
        """Discount node values at step ``i + 1`` back to step ``i``."""
        r = self.rates[i, : i + 1]
        return np.exp(-r * self.dt) * 0.5 * (values[: i + 1] + values[1 : i + 2])

    def bond_values(self, T: float, until: int = 0) -> np.ndarray:
        """Value at each node of step ``until`` of a unit bond maturing at ``T``."""
        n = self._step_index(T)
        values = np.ones(n + 1)
        for i in range(n - 1, until - 1, -1):
            values = self._rollback(values, i)
        return values

    def zero_coupon_bond(self, T: float) -> float:
        """Time-0 price of a unit zero-coupon bond maturing at node date ``T``."""
        return float(self.bond_values(T)[0])

    def bond_option(self, expiry: float, maturity: float, K: float, option_type: str = "call", american: bool = False) -> float:
        """European or American option on a zero-coupon bond, by backward induction.

        Parameters
        ----------
        expiry, maturity : float
            Option expiry and bond maturity; node dates with ``expiry < maturity``.
        K : float
            Strike, per unit face.
        option_type : {"call", "put"}
        american : bool
            Allow exercise at every node up to ``expiry``.

        Returns
        -------
        float
        """
        sign = {"call": 1.0, "put": -1.0}[option_type]
        m = self._step_index(expiry)
        n = self._step_index(maturity)
        if m >= n:
            raise ValueError("expiry must precede the bond's maturity")
        bond = np.ones(n + 1)
        bonds = {n: bond}
        for i in range(n - 1, -1, -1):
            bond = self._rollback(bond, i)
            bonds[i] = bond
        value = np.maximum(sign * (bonds[m] - K), 0.0)
        for i in range(m - 1, -1, -1):
            value = self._rollback(value, i)
            if american:
                value = np.maximum(value, sign * (bonds[i] - K))
        return float(value[0])


@dataclass
class CapFloorResult:
    """A cap or floor as a strip of caplets (floorlets), each priced by Black-76."""

    price: float
    """float: Sum of the caplet prices."""

    caplets: np.ndarray
    """ndarray, shape (n,): Price of each caplet."""

    forwards: np.ndarray
    """ndarray, shape (n,): Simple forward rate of each accrual period."""

    reset_times: np.ndarray
    """ndarray, shape (n,): Fixing dates (option expiries)."""

    payment_times: np.ndarray
    """ndarray, shape (n,): Payment dates (period ends)."""

    strike: float
    """float: Cap or floor rate."""

    kind: str = "cap"
    """str: ``"cap"`` or ``"floor"``."""


@dataclass
class JamshidianResult:
    """An option on a coupon bond split into zero-coupon bond options (Jamshidian 1989)."""

    price: float
    """float: Option value, the sum of :attr:`components`."""

    critical_rate: float
    """float: Short rate :math:`r^*` at expiry at which the bond is worth the strike."""

    payment_times: np.ndarray
    """ndarray, shape (n,): Payment dates of the bond."""

    cash_flows: np.ndarray
    """ndarray, shape (n,): Amounts paid at :attr:`payment_times`."""

    strikes: np.ndarray
    """ndarray, shape (n,): Zero-bond strikes :math:`K_i = P(T, t_i; r^*)`."""

    components: np.ndarray
    """ndarray, shape (n,): ``cash_flows[i]`` times the zero-bond option struck at ``strikes[i]``."""


@dataclass
class HJMResult:
    """Simulated forward curves :math:`f(t, T_j)` under a one-factor HJM model."""

    times: np.ndarray
    """ndarray, shape (n + 1,): Simulation dates, which are also the maturity grid :math:`T_j`."""

    forwards: np.ndarray
    """ndarray, shape (n_paths, n + 1, n + 1): ``forwards[p, i, j] = f(t_i, T_j)``; meaningful for ``j >= i``."""

    initial_curve: np.ndarray
    """ndarray, shape (n + 1,): :math:`f(0, T_j)`."""

    drift_condition: bool = True
    """bool: Whether the HJM no-arbitrage drift was applied."""

    @property
    def short_rates(self) -> np.ndarray:
        """ndarray, shape (n_paths, n + 1): :math:`r(t_i) = f(t_i, t_i)`."""
        idx = np.arange(self.times.size)
        return self.forwards[:, idx, idx]

    def bond_prices(self) -> tuple[np.ndarray, np.ndarray]:
        """Monte Carlo :math:`P(0, T_j) = E[e^{-\\int_0^{T_j} r\\,dt}]` (trapezoidal rule) and its standard errors.

        Returns
        -------
        prices, std_errors : ndarray, shape (n + 1,)
        """
        disc = np.exp(-cumulative_trapezoid(self.short_rates, self.times, axis=1, initial=0.0))
        return disc.mean(axis=0), disc.std(axis=0, ddof=1) / np.sqrt(disc.shape[0])
