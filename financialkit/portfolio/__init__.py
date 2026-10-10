"""financialkit.portfolio: mean-variance optimization, growth-optimal strategies, covariance estimation and risk-based allocation.

Markowitz's efficient frontier (closed form and long-only through
:func:`scipy.optimize.minimize`), Merton's analytic constants, Tobin's
tangency portfolio and two-fund separation; Kelly's criterion, Merton's
portfolio problem and CPPI; Sharpe's single-index model and Ledoit-Wolf
shrinkage; Black-Litterman and Michaud resampling; Rockafellar-Uryasev
CVaR optimization (:func:`scipy.optimize.linprog`), the most-diversified
portfolio, risk parity and hierarchical risk parity; and the Sharpe,
Sortino and information ratios, Grinold's fundamental law and rolling
out-of-sample tests against 1/N. Returns are simple and per period,
weights are fractions of wealth.
"""

from financialkit import __version__
from financialkit.portfolio.core.base import (
    ActiveManagementResult,
    BlackLittermanResult,
    CovarianceResult,
    CPPIResult,
    CVaRResult,
    FrontierConstants,
    FrontierResult,
    HRPResult,
    OutOfSampleResult,
    PortfolioResult,
    ResampledFrontierResult,
)
from financialkit.portfolio.systems.bayesian import black_litterman, implied_equilibrium_returns, resampled_frontier
from financialkit.portfolio.systems.covariance import ledoit_wolf, sample_covariance, single_index_covariance
from financialkit.portfolio.systems.growth import (
    cppi,
    cppi_continuous,
    kelly_fraction,
    kelly_growth_rate,
    kelly_portfolio,
    kelly_wealth_paths,
    merton_certainty_equivalent,
    merton_consumption_rate,
    merton_fraction,
    merton_wealth_paths,
)
from financialkit.portfolio.systems.mean_variance import (
    efficient_frontier,
    efficient_portfolio,
    frontier_constants,
    min_variance_portfolio,
    tangency_portfolio,
    two_fund_allocation,
)
from financialkit.portfolio.systems.out_of_sample import equal_weight, out_of_sample_returns
from financialkit.portfolio.systems.performance import (
    downside_deviation,
    fundamental_law,
    information_ratio,
    sharpe_ratio,
    sharpe_ratio_se,
    simulate_active_management,
    sortino_ratio,
)
from financialkit.portfolio.systems.risk_based import (
    cvar_optimization,
    diversification_ratio,
    hierarchical_risk_parity,
    inverse_volatility_portfolio,
    most_diversified_portfolio,
    risk_parity,
)
from financialkit.portfolio.utils.optimize import portfolio_moments

__all__ = [
    "__version__",
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
    "frontier_constants",
    "min_variance_portfolio",
    "efficient_portfolio",
    "efficient_frontier",
    "tangency_portfolio",
    "two_fund_allocation",
    "equal_weight",
    "out_of_sample_returns",
    "kelly_fraction",
    "kelly_growth_rate",
    "kelly_wealth_paths",
    "kelly_portfolio",
    "merton_fraction",
    "merton_certainty_equivalent",
    "merton_consumption_rate",
    "merton_wealth_paths",
    "cppi",
    "cppi_continuous",
    "sample_covariance",
    "single_index_covariance",
    "ledoit_wolf",
    "implied_equilibrium_returns",
    "black_litterman",
    "resampled_frontier",
    "cvar_optimization",
    "diversification_ratio",
    "most_diversified_portfolio",
    "inverse_volatility_portfolio",
    "risk_parity",
    "hierarchical_risk_parity",
    "sharpe_ratio",
    "sharpe_ratio_se",
    "information_ratio",
    "downside_deviation",
    "sortino_ratio",
    "fundamental_law",
    "simulate_active_management",
    "portfolio_moments",
]
