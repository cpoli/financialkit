"""Abstract base classes and result containers for financialkit.risk.

Sign convention, package-wide in this domain: inputs are profits and
losses (or returns), so a loss is a negative number, and every risk
figure is reported as a positive loss. ``alpha`` is the confidence level
(``0.99`` for a 99% VaR), so the tail probability is ``1 - alpha``.

Closed forms (parametric VaR and ES, Cornish-Fisher, tail-dependence
coefficients) return plain floats or arrays that broadcast. Everything
estimated from scenarios returns a small dataclass that the visualizers
plot. :class:`RiskMeasure` is the interface Artzner, Delbaen, Eber and
Heath's coherence axioms are stated for; VaR and expected shortfall both
implement it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "VaRResult",
    "BacktestResult",
    "ESBacktestResult",
    "TrafficLightResult",
    "GEVResult",
    "GPDResult",
    "CoherenceResult",
    "AllocationResult",
    "FRTBResult",
    "DrawdownResult",
    "RiskMeasure",
]


@dataclass
class VaRResult:
    """Value at risk and expected shortfall estimated from scenarios."""

    var: float
    """Value at risk, a positive loss."""

    es: float
    """Expected shortfall at the same level, a positive loss."""

    alpha: float
    """Confidence level, e.g. ``0.99``."""

    method: str
    """``"historical"``, ``"hybrid"``, ``"filtered_historical"`` or ``"monte_carlo"``."""

    scenarios: np.ndarray
    """ndarray, shape (n,): Scenario profits and losses the figures were read from."""

    weights: np.ndarray
    """ndarray, shape (n,): Probability of each scenario (equal unless age-weighted)."""

    ci: tuple[float, float] = (float("nan"), float("nan"))
    """Order-statistic confidence interval for the VaR (Monte Carlo only)."""


@dataclass
class BacktestResult:
    """A likelihood-ratio test of VaR exceptions."""

    statistic: float
    """Likelihood-ratio statistic."""

    p_value: float
    """Asymptotic chi-squared p-value."""

    df: int
    """Degrees of freedom of the chi-squared reference distribution."""

    n_exceptions: int
    """Number of days the loss exceeded the VaR."""

    n_obs: int
    """Number of days tested."""

    expected: float
    """Expected number of exceptions, ``n_obs * (1 - alpha)``."""

    def reject(self, level: float = 0.05) -> bool:
        """Whether the model is rejected at significance ``level``."""
        return bool(self.p_value < level)


@dataclass
class ESBacktestResult:
    """An Acerbi-Szekely backtest of expected-shortfall forecasts."""

    statistic: float
    """Observed test statistic; zero is its expectation under a correct model, negative values mean risk was underestimated."""

    p_value: float
    """Fraction of statistics simulated under the model at or below the observed one (``nan`` without simulations)."""

    test: str
    """``"z1"`` (size of the exceptions given their number) or ``"z2"`` (number and size jointly)."""

    n_exceptions: int
    """Number of days the loss exceeded the VaR."""

    n_obs: int
    """Number of days tested."""

    expected: float
    """Expected number of exceptions, ``n_obs * (1 - alpha)``."""

    def reject(self, level: float = 0.05) -> bool:
        """Whether the model is rejected at significance ``level``."""
        return bool(self.p_value < level)


@dataclass
class TrafficLightResult:
    """The Basel Committee's (1996) backtesting zone for an internal model."""

    zone: str
    """``"green"``, ``"yellow"`` or ``"red"``."""

    n_exceptions: int
    """Exceptions observed."""

    cumulative_probability: float
    """Binomial probability of at most ``n_exceptions`` exceptions for an accurate model."""

    multiplier: float
    """Capital multiplier, 3 plus the plus factor (``nan`` outside the 250-day, 99% setting)."""


@dataclass
class GEVResult:
    """A generalized extreme value fit to block maxima of losses."""

    xi: float
    """Shape: ``> 0`` Fréchet (fat tails), ``0`` Gumbel, ``< 0`` Weibull."""

    mu: float
    """Location."""

    sigma: float
    """Scale."""

    block_size: int
    """Observations per block."""

    maxima: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (n_blocks,): The block maxima of the losses."""


@dataclass
class GPDResult:
    """A generalized Pareto fit to losses above a threshold."""

    xi: float
    """Tail index: ``1 / xi`` is the number of finite moments."""

    beta: float
    """Scale of the excesses."""

    threshold: float
    """Loss threshold ``u``."""

    n_exceed: int
    """Number of losses above ``u``."""

    n_obs: int
    """Total number of observations."""

    excesses: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (n_exceed,): Losses above ``u`` minus ``u``."""


@dataclass
class CoherenceResult:
    """Which of Artzner et al.'s four axioms a risk measure satisfied on a test pair."""

    monotonicity: bool
    """``rho(Z) >= rho(X)`` for a position ``Z <= X`` in every state."""

    subadditivity: bool
    """``rho(X + Y) <= rho(X) + rho(Y)``."""

    positive_homogeneity: bool
    """``rho(lam X) = lam rho(X)``."""

    translation_invariance: bool
    """``rho(X + c) = rho(X) - c``."""

    rho_x: float
    """``rho(X)``."""

    rho_y: float
    """``rho(Y)``."""

    rho_sum: float
    """``rho(X + Y)``."""

    @property
    def coherent(self) -> bool:
        """True when all four axioms held."""
        return self.monotonicity and self.subadditivity and self.positive_homogeneity and self.translation_invariance


@dataclass
class AllocationResult:
    """Euler allocation of a portfolio risk figure to its positions."""

    contributions: np.ndarray
    """ndarray, shape (d,): Risk contribution of each position; they sum to ``total``."""

    total: float
    """Portfolio risk figure."""

    measure: str
    """``"var"``, ``"es"`` or ``"volatility"``."""

    @property
    def shares(self) -> np.ndarray:
        """ndarray: Contributions as fractions of the total."""
        return self.contributions / self.total


@dataclass
class FRTBResult:
    """The FRTB liquidity-horizon cascade of ten-day expected shortfalls."""

    es: float
    """Liquidity-adjusted ES, the root sum of squares of ``scaled``."""

    es_by_horizon: np.ndarray
    """ndarray, shape (5,): Ten-day ES with only the factors of horizon at least ``horizons[j]`` shocked."""

    horizons: np.ndarray
    """ndarray, shape (5,): The liquidity horizons in days, 10, 20, 40, 60 and 120."""

    scaled: np.ndarray
    """ndarray, shape (5,): ``es_by_horizon[j] * sqrt((horizons[j] - horizons[j - 1]) / 10)``, the horizon before the first taken as 0."""

    alpha: float
    """Confidence level, 0.975 in the standard."""


@dataclass
class DrawdownResult:
    """The drawdown path of a wealth index and its summary statistics."""

    wealth: np.ndarray
    """ndarray, shape (n + 1,): Wealth index, starting at 1."""

    peak: np.ndarray
    """ndarray, shape (n + 1,): Running maximum of the wealth."""

    drawdown: np.ndarray
    """ndarray, shape (n + 1,): Fractional fall from the peak, ``1 - wealth / peak`` (non-negative)."""

    max_drawdown: float
    """Largest drawdown."""

    peak_index: int
    """Index of the peak preceding the largest drawdown."""

    trough_index: int
    """Index of the trough of the largest drawdown."""

    recovery_index: int
    """First index after the trough at which the peak is regained (``-1`` if never)."""

    @property
    def longest_duration(self) -> int:
        """int: Longest run of consecutive periods spent below a previous peak."""
        under = self.drawdown > 0
        best = run = 0
        for u in under:
            run = run + 1 if u else 0
            best = max(best, run)
        return best


class RiskMeasure(ABC):
    """A map from a random profit and loss, given by scenarios, to a capital figure.

    Parameters
    ----------
    alpha : float
        Confidence level in ``(0, 1)``.
    """

    name: str = ""

    def __init__(self, alpha: float = 0.99):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        self.alpha = float(alpha)

    @abstractmethod
    def __call__(self, pnl: ArrayLike) -> float:
        """Risk of the equally likely scenario profits and losses ``pnl`` (a positive loss)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha={self.alpha})"
