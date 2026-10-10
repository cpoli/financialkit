"""Abstract base classes and result containers for financialkit.asset_pricing.

Conventions in this domain: returns are simple per-period returns;
``excess_returns`` are returns minus the risk-free rate; panels of
returns have shape ``(n_periods, n_assets)``. The consumption-based
models and the stochastic discount factor work with *gross* returns
:math:`R = 1 + r`, so that the pricing equation reads
:math:`E[m R] = 1`.

Closed forms (Gordon, CAPM and APT expected returns, Modigliani-Miller)
return plain floats or arrays that broadcast. Estimators and equilibria
return a small dataclass that the visualizers plot.
:class:`ConsumptionEconomy` is the interface for the Lucas-tree
economies: given a time-discount factor and a risk aversion, each one
prices the consumption claim and the riskless bond and reports the
moments of its stochastic discount factor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CAPMEquilibrium",
    "ZeroBetaResult",
    "FactorRegressionResult",
    "GRSResult",
    "FamaMacBethResult",
    "ExcessVolatilityResult",
    "ConsumptionMoments",
    "EconomyPath",
    "HJBound",
    "ConsumptionEconomy",
]


@dataclass
class CAPMEquilibrium:
    """The CAPM equilibrium of a mean-variance market with a riskless asset."""

    market_weights: np.ndarray
    """ndarray, shape (d,): Tangency (market) portfolio, summing to one."""

    betas: np.ndarray
    """ndarray, shape (d,): Beta of each asset on the market portfolio."""

    expected_returns: np.ndarray
    """ndarray, shape (d,): Expected returns of the assets."""

    risk_free: float
    """Riskless rate."""

    market_return: float
    """Expected return of the market portfolio."""

    market_vol: float
    """Volatility of the market portfolio."""

    @property
    def market_premium(self) -> float:
        """float: Expected market return minus the riskless rate."""
        return self.market_return - self.risk_free

    @property
    def sml_errors(self) -> np.ndarray:
        """ndarray: Distance of each asset from the security market line (zero in equilibrium)."""
        return self.expected_returns - (self.risk_free + self.betas * self.market_premium)


@dataclass
class ZeroBetaResult:
    """Black's zero-beta CAPM for an efficient market portfolio, without a riskless asset."""

    market_weights: np.ndarray
    """ndarray, shape (d,): Efficient (market) portfolio."""

    zero_beta_weights: np.ndarray
    """ndarray, shape (d,): Minimum-variance portfolio uncorrelated with the market."""

    zero_beta_rate: float
    """Expected return of the zero-beta portfolio: the security market line's intercept."""

    betas: np.ndarray
    """ndarray, shape (d,): Beta of each asset on the market portfolio."""

    expected_returns: np.ndarray
    """ndarray, shape (d,): Expected returns of the assets."""

    market_return: float
    """Expected return of the market portfolio."""


@dataclass
class FactorRegressionResult:
    """Time-series regressions :math:`r^e_{it} = \\alpha_i + \\beta_i' f_t + \\varepsilon_{it}`, one per asset."""

    alphas: np.ndarray
    """ndarray, shape (n,): Intercepts, the pricing errors."""

    betas: np.ndarray
    """ndarray, shape (n, k): Factor loadings."""

    alpha_se: np.ndarray
    """ndarray, shape (n,): Classical standard errors of the intercepts."""

    r_squared: np.ndarray
    """ndarray, shape (n,): Explained share of each asset's variance."""

    residuals: np.ndarray
    """ndarray, shape (T, n): Regression residuals."""

    @property
    def alpha_t(self) -> np.ndarray:
        """ndarray: t-statistics of the intercepts."""
        return self.alphas / self.alpha_se


@dataclass
class GRSResult:
    """Gibbons, Ross and Shanken's joint F-test that every alpha is zero."""

    statistic: float
    """F-statistic."""

    p_value: float
    """Exact p-value under normal errors."""

    df1: int
    """Numerator degrees of freedom, the number of test assets."""

    df2: int
    """Denominator degrees of freedom, ``T - N - K``."""

    alphas: np.ndarray
    """ndarray, shape (n,): Estimated intercepts."""

    def reject(self, level: float = 0.05) -> bool:
        """Whether the model is rejected at significance ``level``."""
        return bool(self.p_value < level)


@dataclass
class FamaMacBethResult:
    """Fama-MacBeth cross-sectional regressions :math:`r^e_{it} = \\gamma_{0t} + \\gamma_t' \\beta_i + u_{it}`."""

    gammas: np.ndarray
    """ndarray, shape (T, k + 1): Each period's intercept and slopes."""

    premia: np.ndarray
    """ndarray, shape (k + 1,): Time averages of the gammas: the zero-beta excess return, then the factor premia."""

    std_errors: np.ndarray
    """ndarray, shape (k + 1,): Standard errors from the time series of gammas, ``std / sqrt(T)``."""

    betas: np.ndarray
    """ndarray, shape (n, k): Loadings used as regressors in the second pass."""

    @property
    def t_values(self) -> np.ndarray:
        """ndarray: Premia divided by their standard errors."""
        return self.premia / self.std_errors


@dataclass
class ExcessVolatilityResult:
    """Shiller's variance-bound comparison of actual and ex-post rational prices."""

    price: np.ndarray
    """ndarray, shape (T,): Actual price."""

    rational_price: np.ndarray
    """ndarray, shape (T,): Ex-post rational price :math:`p^*`, the discounted realized dividends."""

    std_price: float
    """Standard deviation of the actual price."""

    std_rational: float
    """Standard deviation of :math:`p^*`."""

    @property
    def ratio(self) -> float:
        """float: ``std_price / std_rational``; above one violates the bound."""
        return self.std_price / self.std_rational

    @property
    def violated(self) -> bool:
        """bool: Whether the price is more volatile than the efficient-markets bound allows."""
        return bool(self.ratio > 1.0)


@dataclass
class ConsumptionMoments:
    """Unconditional moments of a consumption-based economy; all returns are gross."""

    price_dividend: float
    """Average price-dividend ratio of the consumption claim."""

    risk_free: float
    """Average gross riskless return."""

    equity_return: float
    """Expected gross return of the consumption claim."""

    equity_vol: float
    """Standard deviation of the equity return."""

    sdf_mean: float
    """:math:`E[m]`."""

    sdf_vol: float
    """:math:`\\sigma(m)`."""

    @property
    def equity_premium(self) -> float:
        """float: Expected equity return minus the riskless return."""
        return self.equity_return - self.risk_free

    @property
    def sharpe_ratio(self) -> float:
        """float: Equity premium over equity volatility."""
        return self.equity_premium / self.equity_vol


@dataclass
class EconomyPath:
    """A simulated path of a consumption economy; returns are gross and dated at the period's end."""

    growth: np.ndarray
    """ndarray, shape (n,): Gross consumption (and dividend) growth."""

    sdf: np.ndarray
    """ndarray, shape (n,): Stochastic discount factor :math:`\\beta G^{-\\gamma}`."""

    equity_return: np.ndarray
    """ndarray, shape (n,): Gross return of the consumption claim."""

    risk_free: np.ndarray
    """ndarray, shape (n,): Gross riskless return, known at the start of the period."""


@dataclass
class HJBound:
    """Hansen and Jagannathan's lower bound on the volatility of a stochastic discount factor."""

    sdf_means: np.ndarray
    """ndarray, shape (m,): Candidate means :math:`E[m]`."""

    sdf_stds: np.ndarray
    """ndarray, shape (m,): Smallest :math:`\\sigma(m)` that prices the assets at each mean."""

    def admits(self, sdf_mean: float, sdf_std: float) -> bool:
        """Whether an SDF with these moments lies inside the admissible region."""
        return bool(sdf_std >= np.interp(sdf_mean, self.sdf_means, self.sdf_stds))


class ConsumptionEconomy(ABC):
    """An endowment economy with a CRRA representative agent who eats the dividend of a Lucas tree.

    Parameters
    ----------
    beta : float
        Time-discount factor in ``(0, 1]``.
    gamma : float
        Relative risk aversion, positive.
    """

    def __init__(self, beta: float, gamma: float):
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must lie in (0, 1], got {beta}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.beta = float(beta)
        self.gamma = float(gamma)

    @abstractmethod
    def moments(self) -> ConsumptionMoments:
        """Unconditional moments of the riskless rate, the equity return and the SDF."""

    @abstractmethod
    def simulate(self, n_periods: int, rng: np.random.Generator | int | None = None) -> EconomyPath:
        """Simulate consumption growth with the matching SDF and returns."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(beta={self.beta}, gamma={self.gamma})"
