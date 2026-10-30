"""financialkit.asset_pricing: valuation, the CAPM, factor models, consumption-based pricing and the SDF.

Williams's dividend discount model and Gordon growth (through
:func:`financialkit.time_value.perpetuity_present_value`),
Modigliani-Miller and the cost of capital, Shiller's variance bound;
Sharpe's CAPM with the market as the tangency portfolio, Jensen's alpha,
Black's zero-beta CAPM and the Gibbons-Ross-Shanken test; time-series
factor regressions (through :func:`financialkit.time_series.ols`),
Fama-MacBeth premia, Ross's APT and arbitrage portfolios, and the
characteristic sorts behind Fama-French and Carhart factors, the
five-factor model and multiple-testing hurdles for the factor zoo;
Lucas's tree, Breeden's consumption CAPM, Mehra-Prescott's equity
premium puzzle, Campbell-Cochrane habit and Bansal-Yaron long-run risk;
Harrison-Kreps state prices and the Hansen-Jagannathan bound; Hansen's
GMM with Newey-West weighting and the Hansen-Singleton Euler equation;
and the Campbell-Shiller log-linear present value.
Return panels have shape ``(n_periods, n_assets)``, and the
consumption-based models use gross returns.
"""

from financialkit import __version__
from financialkit.asset_pricing.core.base import (
    CAPMEquilibrium,
    ConsumptionEconomy,
    ConsumptionMoments,
    EconomyPath,
    ExcessVolatilityResult,
    FactorRegressionResult,
    FamaMacBethResult,
    GMMResult,
    GRSResult,
    HJBound,
    PresentValueDecomposition,
    ZeroBetaResult,
)
from financialkit.asset_pricing.systems.capm import capm_equilibrium, capm_expected_return, estimate_beta, grs_test, jensen_alpha, zero_beta_capm
from financialkit.asset_pricing.systems.consumption import (
    BansalYaron,
    CampbellCochrane,
    LucasTree,
    MehraPrescott,
    ccapm_expected_excess_return,
    hansen_singleton,
    implied_risk_aversion,
)
from financialkit.asset_pricing.systems.factors import (
    adjusted_p_values,
    apt_arbitrage_portfolio,
    apt_expected_returns,
    fama_french_five,
    fama_macbeth,
    t_hurdle,
    time_series_regression,
)
from financialkit.asset_pricing.systems.gmm import gmm
from financialkit.asset_pricing.systems.sdf import FiniteStateMarket, hansen_jagannathan_bound, max_sharpe_ratio
from financialkit.asset_pricing.systems.valuation import (
    ar1_dividend_price,
    campbell_shiller_decomposition,
    campbell_shiller_rho,
    dividend_discount_value,
    ex_post_rational_price,
    gordon_growth_price,
    implied_cost_of_equity,
    levered_cost_of_equity,
    levered_firm_value,
    log_linear_return,
    sustainable_growth,
    variance_bound,
    wacc,
)
from financialkit.asset_pricing.utils.hac import newey_west
from financialkit.asset_pricing.utils.sorts import long_short_factor, momentum_signal, sort_portfolios, two_by_three_portfolios
from financialkit.asset_pricing.utils.synthetic import FactorPanel, simulate_factor_model
from financialkit.asset_pricing.visualizers.plots import (
    plot_alphas,
    plot_excess_volatility,
    plot_fama_macbeth,
    plot_hj_bound,
    plot_sml,
)

__all__ = [
    "__version__",
    "CAPMEquilibrium",
    "ZeroBetaResult",
    "FactorRegressionResult",
    "GRSResult",
    "FamaMacBethResult",
    "ExcessVolatilityResult",
    "ConsumptionMoments",
    "EconomyPath",
    "HJBound",
    "GMMResult",
    "PresentValueDecomposition",
    "ConsumptionEconomy",
    "dividend_discount_value",
    "gordon_growth_price",
    "implied_cost_of_equity",
    "sustainable_growth",
    "levered_cost_of_equity",
    "levered_firm_value",
    "wacc",
    "ex_post_rational_price",
    "ar1_dividend_price",
    "variance_bound",
    "campbell_shiller_rho",
    "log_linear_return",
    "campbell_shiller_decomposition",
    "capm_expected_return",
    "estimate_beta",
    "capm_equilibrium",
    "jensen_alpha",
    "zero_beta_capm",
    "grs_test",
    "time_series_regression",
    "fama_macbeth",
    "apt_expected_returns",
    "apt_arbitrage_portfolio",
    "fama_french_five",
    "adjusted_p_values",
    "t_hurdle",
    "LucasTree",
    "MehraPrescott",
    "CampbellCochrane",
    "BansalYaron",
    "ccapm_expected_excess_return",
    "implied_risk_aversion",
    "hansen_singleton",
    "gmm",
    "FiniteStateMarket",
    "hansen_jagannathan_bound",
    "max_sharpe_ratio",
    "sort_portfolios",
    "long_short_factor",
    "momentum_signal",
    "two_by_three_portfolios",
    "newey_west",
    "FactorPanel",
    "simulate_factor_model",
    "plot_sml",
    "plot_alphas",
    "plot_fama_macbeth",
    "plot_excess_volatility",
    "plot_hj_bound",
]
