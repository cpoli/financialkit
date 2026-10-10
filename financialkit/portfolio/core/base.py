"""Result containers for financialkit.portfolio.

Conventions, package-wide in this domain: ``mu`` is a vector of expected
one-period returns, ``cov`` their covariance matrix and ``returns`` a
``(n_periods, n_assets)`` array of simple returns, one asset per column.
Weights are fractions of wealth and sum to one, except in
:func:`~financialkit.portfolio.systems.mean_variance.two_fund_allocation`
and the growth-optimal rules, where whatever the risky weights leave over
sits in the riskless asset.

Every optimizer returns a small dataclass that the visualizers plot;
closed forms (the Kelly fraction, Merton's fraction, the Sharpe ratio)
return plain floats or arrays that broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "PortfolioResult",
    "FrontierResult",
    "FrontierConstants",
    "CovarianceResult",
    "BlackLittermanResult",
    "ResampledFrontierResult",
    "CVaRResult",
    "HRPResult",
    "CPPIResult",
    "ActiveManagementResult",
    "OutOfSampleResult",
]


@dataclass
class PortfolioResult:
    """One portfolio and its first two moments."""

    weights: np.ndarray
    """ndarray, shape (d,): Weight of each asset."""

    expected_return: float
    """Expected return ``w @ mu`` (``nan`` when no ``mu`` was given)."""

    volatility: float
    """Standard deviation ``sqrt(w @ cov @ w)``."""

    method: str
    """Name of the rule that produced the weights."""

    def sharpe_ratio(self, rf: float = 0.0) -> float:
        """Excess return per unit of volatility."""
        return float((self.expected_return - rf) / self.volatility)


@dataclass
class FrontierResult:
    """Points along a mean-variance efficient frontier."""

    returns: np.ndarray
    """ndarray, shape (n_points,): Target expected returns."""

    volatilities: np.ndarray
    """ndarray, shape (n_points,): Minimum volatility reaching each target."""

    weights: np.ndarray
    """ndarray, shape (n_points, d): The efficient portfolio at each target."""

    mu: np.ndarray
    """ndarray, shape (d,): Expected returns of the assets."""

    cov: np.ndarray
    """ndarray, shape (d, d): Covariance of the assets."""

    long_only: bool
    """Whether short sales were excluded."""

    @property
    def asset_volatilities(self) -> np.ndarray:
        """ndarray, shape (d,): Volatility of each asset on its own."""
        return np.sqrt(np.diag(self.cov))


@dataclass
class FrontierConstants:
    """Merton's (1972) four scalars that pin down the unconstrained frontier.

    With :math:`A = \\mathbf 1'\\Sigma^{-1}\\mu`,
    :math:`B = \\mu'\\Sigma^{-1}\\mu`, :math:`C = \\mathbf 1'\\Sigma^{-1}\\mathbf 1`
    and :math:`D = BC - A^2`, the least variance at mean :math:`m` is
    :math:`\\sigma^2(m) = (C m^2 - 2 A m + B) / D`, a hyperbola in the
    mean-volatility plane.
    """

    A: float
    B: float
    C: float
    D: float

    def variance(self, m):
        """Least portfolio variance with expected return ``m`` (broadcasts)."""
        m = np.asarray(m, dtype=np.float64)
        out = (self.C * m**2 - 2.0 * self.A * m + self.B) / self.D
        return float(out) if out.ndim == 0 else out

    def volatility(self, m):
        """Least portfolio volatility with expected return ``m`` (broadcasts)."""
        return np.sqrt(self.variance(m))

    @property
    def gmv_return(self) -> float:
        """Expected return of the global minimum-variance portfolio, ``A / C``."""
        return self.A / self.C

    @property
    def gmv_volatility(self) -> float:
        """Volatility of the global minimum-variance portfolio, ``1 / sqrt(C)``."""
        return float(1.0 / np.sqrt(self.C))

    def asymptote_slope(self) -> float:
        """Slope ``sqrt(D / C)`` of the hyperbola's asymptotes."""
        return float(np.sqrt(self.D / self.C))


@dataclass
class CovarianceResult:
    """An estimated covariance matrix."""

    cov: np.ndarray
    """ndarray, shape (d, d): The estimate."""

    method: str
    """``"sample"``, ``"single_index"`` or ``"ledoit_wolf"``."""

    n_params: int
    """Number of free parameters the estimator fits."""

    shrinkage: float = 0.0
    """Weight on the shrinkage target (Ledoit-Wolf only)."""

    target: Optional[np.ndarray] = None
    """ndarray, shape (d, d): Shrinkage target, or the single-index model's structured part."""

    betas: Optional[np.ndarray] = None
    """ndarray, shape (d,): Market betas (single-index model only)."""

    @property
    def corr(self) -> np.ndarray:
        """ndarray, shape (d, d): The implied correlation matrix."""
        s = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(s, s)

    @property
    def condition_number(self) -> float:
        """Ratio of the largest to the smallest eigenvalue."""
        ev = np.linalg.eigvalsh(self.cov)
        return float(ev[-1] / ev[0])


@dataclass
class BlackLittermanResult:
    """Black and Litterman's blend of market equilibrium and investor views."""

    prior_returns: np.ndarray
    """ndarray, shape (d,): Implied equilibrium returns :math:`\\Pi = \\delta \\Sigma w_{mkt}`."""

    posterior_returns: np.ndarray
    """ndarray, shape (d,): Expected returns after the views."""

    posterior_cov: np.ndarray
    """ndarray, shape (d, d): Covariance of returns after the views, :math:`\\Sigma + M`."""

    weights: np.ndarray
    """ndarray, shape (d,): Unconstrained optimal weights :math:`(\\delta(\\Sigma + M))^{-1} \\mu_{BL}`."""

    market_weights: np.ndarray
    """ndarray, shape (d,): Capitalisation weights the prior is reverse-engineered from."""

    tau: float
    """Uncertainty of the prior relative to returns."""

    delta: float
    """Market risk aversion."""


@dataclass
class ResampledFrontierResult:
    """Michaud's resampled frontier: rank-averaged weights from resampled inputs."""

    weights: np.ndarray
    """ndarray, shape (n_points, d): Averaged weights at each frontier rank."""

    sample_weights: np.ndarray
    """ndarray, shape (n_resamples, n_points, d): The frontier of every resample."""

    returns: np.ndarray
    """ndarray, shape (n_points,): Expected return of each averaged portfolio under the true inputs."""

    volatilities: np.ndarray
    """ndarray, shape (n_points,): Volatility of each averaged portfolio under the true inputs."""

    frontier: FrontierResult
    """The classical frontier of the true inputs, for comparison."""


@dataclass
class CVaRResult:
    """A portfolio minimizing conditional value at risk over scenarios."""

    weights: np.ndarray
    """ndarray, shape (d,): Optimal weights."""

    var: float
    """Value at risk of the optimal portfolio, the LP's auxiliary variable :math:`\\zeta` (a positive loss)."""

    cvar: float
    """Minimized conditional value at risk (a positive loss)."""

    expected_return: float
    """Mean scenario return of the optimal portfolio."""

    alpha: float
    """Confidence level."""


@dataclass
class HRPResult:
    """Lopez de Prado's hierarchical risk parity allocation."""

    weights: np.ndarray
    """ndarray, shape (d,): Weights in the original asset order."""

    order: np.ndarray
    """ndarray, shape (d,): Asset indices in quasi-diagonal (dendrogram leaf) order."""

    linkage: np.ndarray
    """ndarray, shape (d - 1, 4): :func:`scipy.cluster.hierarchy.linkage` matrix of the correlation distance."""


@dataclass
class CPPIResult:
    """Wealth paths of a constant proportion portfolio insurance strategy."""

    times: np.ndarray
    """ndarray, shape (n_steps + 1,): Rebalancing dates."""

    prices: np.ndarray
    """ndarray, shape (n_paths, n_steps + 1): Risky asset prices."""

    wealth: np.ndarray
    """ndarray, shape (n_paths, n_steps + 1): Portfolio value."""

    floor: np.ndarray
    """ndarray, shape (n_steps + 1,): Floor, growing at the riskless rate."""

    exposure: np.ndarray
    """ndarray, shape (n_paths, n_steps): Amount held in the risky asset over each step."""

    multiplier: float
    """Exposure per unit of cushion."""

    @property
    def cushion(self) -> np.ndarray:
        """ndarray, shape (n_paths, n_steps + 1): Wealth above the floor."""
        return self.wealth - self.floor

    @property
    def breached(self) -> np.ndarray:
        """ndarray of bool, shape (n_paths,): Paths whose wealth ever fell below the floor (gap risk)."""
        return np.any(self.cushion < -1e-12 * self.floor, axis=1)


@dataclass
class ActiveManagementResult:
    """A simulated active manager, for checking Grinold's fundamental law."""

    active_returns: np.ndarray
    """ndarray, shape (n_periods,): Active return of each period, scaled to unit predicted IR."""

    ir: float
    """Realized information ratio per period."""

    ir_se: float
    """Standard error of ``ir``."""

    predicted_ir: float
    """Grinold's :math:`IC \\sqrt{BR}`."""

    realized_ic: float
    """Average cross-sectional correlation of forecasts and outcomes."""

    breadth: int
    """Independent forecasts per period."""


@dataclass
class OutOfSampleResult:
    """Returns of a portfolio rule re-estimated on a rolling window."""

    returns: np.ndarray
    """ndarray, shape (n_periods - window,): Realized out-of-sample returns."""

    weights: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    """ndarray, shape (n_periods - window, d): Weights held in each period."""

    name: str = ""
    """Label of the rule."""

    @property
    def mean(self) -> float:
        """Mean out-of-sample return per period."""
        return float(self.returns.mean())

    @property
    def sharpe(self) -> float:
        """Out-of-sample Sharpe ratio per period (no riskless rate)."""
        return float(self.returns.mean() / self.returns.std(ddof=1))

    @property
    def turnover(self) -> float:
        """Average sum of absolute weight changes per rebalance."""
        return float(np.abs(np.diff(self.weights, axis=0)).sum(axis=1).mean()) if len(self.weights) > 1 else 0.0
