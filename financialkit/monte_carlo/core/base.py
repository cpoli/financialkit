"""Result containers for financialkit.monte_carlo.

Every estimator in this domain returns a :class:`MonteCarloResult`: the
estimate, its standard error, the number of function evaluations it
cost and a normal-theory confidence interval. Estimators whose samples
are not independent (Latin hypercube, moment matching, randomized
quasi-Monte Carlo) estimate the standard error from independent
replications, so every result can be compared with every other on the
same footing: :meth:`MonteCarloResult.variance_ratio` is the variance
reduction factor per function evaluation.

Families with more structure extend it: a stratified estimate carries its
allocation and the per-stratum statistics, a control-variate estimate its
regression coefficients, a multilevel estimate its level-by-level means,
variances and sample sizes. Monte Carlo Greeks are a separate
:class:`MonteCarloGreeks` container with a standard error per Greek.
These are dataclasses, never bare tuples, so the visualizers have a
stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

__all__ = [
    "MonteCarloResult",
    "StratifiedResult",
    "ControlVariateResult",
    "MonteCarloGreeks",
    "MLMCResult",
]


@dataclass
class MonteCarloResult:
    """A Monte Carlo estimate with its standard error and confidence interval."""

    estimate: float
    """float: The estimate of the expectation or integral."""

    std_error: float
    """float: Standard error of :attr:`estimate`."""

    n_samples: int
    """int: Number of integrand (or payoff) evaluations used."""

    confidence: float = 0.95
    """float: Coverage of :attr:`ci`."""

    method: str = "crude"
    """str: Name of the estimator, for labels."""

    @property
    def half_width(self) -> float:
        """float: Half-width :math:`z_{(1+c)/2}\\,\\mathrm{SE}` of the confidence interval."""
        return float(norm.ppf(0.5 + 0.5 * self.confidence) * self.std_error)

    @property
    def ci(self) -> tuple[float, float]:
        """tuple of float: The normal-theory confidence interval ``(lower, upper)``."""
        return self.estimate - self.half_width, self.estimate + self.half_width

    @property
    def variance(self) -> float:
        """float: Variance per evaluation, :math:`n\\,\\mathrm{SE}^2`; the cost-adjusted figure of merit."""
        return float(self.n_samples * self.std_error**2)

    def covers(self, value: float) -> bool:
        """Whether `value` lies inside :attr:`ci`."""
        lo, hi = self.ci
        return bool(lo <= value <= hi)

    def variance_ratio(self, baseline: MonteCarloResult) -> float:
        """Variance reduction factor relative to `baseline`, per function evaluation.

        Above 1 means this estimator needs that many times fewer
        evaluations than `baseline` for the same standard error.
        """
        return baseline.variance / self.variance


@dataclass
class StratifiedResult(MonteCarloResult):
    """A stratified estimate over equal-probability strata of :math:`[0, 1)` (Neyman 1934)."""

    allocation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    """ndarray of int, shape (H,): Samples drawn in each stratum."""

    stratum_means: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (H,): Sample mean of the integrand in each stratum."""

    stratum_stds: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (H,): Sample standard deviation in each stratum."""


@dataclass
class ControlVariateResult(MonteCarloResult):
    """A control-variate estimate :math:`\\bar Y - \\beta^\\top(\\bar X - \\mu_X)` (Lavenberg-Welch 1981)."""

    beta: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (k,): Estimated regression coefficients of the controls."""

    r_squared: float = 0.0
    """float: Fraction of the variance of ``Y`` explained by the controls; the variance falls by ``1 - r_squared``."""

    crude_estimate: float = float("nan")
    """float: The uncontrolled sample mean of ``Y``."""

    crude_std_error: float = float("nan")
    """float: Its standard error."""


@dataclass
class MonteCarloGreeks:
    """Monte Carlo estimates of an option's price, delta, vega and gamma, each with a standard error.

    A Greek an estimator cannot produce (for example a pathwise gamma of a
    payoff with a kink) is ``nan``.
    """

    price: float
    """float: Discounted mean payoff."""

    delta: float
    """float: :math:`\\partial V / \\partial S`."""

    vega: float
    """float: :math:`\\partial V / \\partial \\sigma`, per unit of volatility."""

    gamma: float
    """float: :math:`\\partial^2 V / \\partial S^2`."""

    price_se: float
    """float: Standard error of :attr:`price`."""

    delta_se: float
    """float: Standard error of :attr:`delta`."""

    vega_se: float
    """float: Standard error of :attr:`vega`."""

    gamma_se: float
    """float: Standard error of :attr:`gamma`."""

    n_paths: int
    """int: Number of simulated terminal prices."""

    method: str = ""
    """str: ``"likelihood_ratio"`` or ``"pathwise"``."""


@dataclass
class MLMCResult(MonteCarloResult):
    """A multilevel Monte Carlo estimate :math:`\\sum_l E[P_l - P_{l-1}]` (Giles 2008).

    Level ``l`` uses ``M**l`` time steps; level 0 estimates :math:`E[P_0]`
    itself. :attr:`cost` counts fine-grid time steps, the work unit of
    Giles's complexity theorem, and :attr:`n_samples` the paths over all
    levels.
    """

    level_means: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (L + 1,): Sample mean of :math:`P_l - P_{l-1}` on each level."""

    level_variances: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (L + 1,): Sample variance of :math:`P_l - P_{l-1}` on each level."""

    level_samples: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    """ndarray of int, shape (L + 1,): Paths simulated on each level."""

    level_costs: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray, shape (L + 1,): Time steps per sample on each level (fine plus coarse)."""

    cost: float = 0.0
    """float: Total time steps simulated, :math:`\\sum_l N_l C_l`."""

    target_rmse: float = float("nan")
    """float: The requested root-mean-square error :math:`\\varepsilon`."""

    M: int = 2
    """int: Refinement factor between levels."""

    @property
    def levels(self) -> np.ndarray:
        """ndarray of int: ``0, 1, ..., L``."""
        return np.arange(self.level_means.size)

    @property
    def standard_mc_cost(self) -> float:
        """float: Time steps standard Monte Carlo on the finest level would need for the same variance.

        That is :math:`V_L^{\\text{fine}} C_L / (\\varepsilon^2 / 2)`, with the
        finest level's payoff variance approximated by the level-0
        variance (the payoff variance barely depends on the step size).
        """
        L = self.level_means.size - 1
        return float(self.level_variances[0] * self.M**L / (0.5 * self.target_rmse**2))
