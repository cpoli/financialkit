"""financialkit.asset_pricing: valuation, the CAPM, factor models, consumption-based pricing and the SDF.

Williams's dividend discount model and Gordon growth (through
:func:`financialkit.time_value.perpetuity_present_value`),
Modigliani-Miller and the cost of capital, Shiller's variance bound;
Sharpe's CAPM with the market as the tangency portfolio, Jensen's alpha,
Black's zero-beta CAPM and the Gibbons-Ross-Shanken test; time-series
factor regressions (through :func:`financialkit.time_series.ols`),
Fama-MacBeth premia, Ross's APT and arbitrage portfolios, and the
characteristic sorts behind Fama-French and Carhart factors; Lucas's
tree, Breeden's consumption CAPM and Mehra-Prescott's equity premium
puzzle; Harrison-Kreps state prices and the Hansen-Jagannathan bound.
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
    GRSResult,
    HJBound,
    ZeroBetaResult,
)
from financialkit.asset_pricing.systems.capm import capm_equilibrium, capm_expected_return, estimate_beta, grs_test, jensen_alpha, zero_beta_capm
from financialkit.asset_pricing.systems.consumption import LucasTree, MehraPrescott, ccapm_expected_excess_return, implied_risk_aversion
from financialkit.asset_pricing.systems.factors import apt_arbitrage_portfolio, apt_expected_returns, fama_macbeth, time_series_regression
from financialkit.asset_pricing.systems.sdf import FiniteStateMarket, hansen_jagannathan_bound, max_sharpe_ratio
from financialkit.asset_pricing.systems.valuation import (
    ar1_dividend_price,
    dividend_discount_value,
    ex_post_rational_price,
    gordon_growth_price,
    implied_cost_of_equity,
    levered_cost_of_equity,
    levered_firm_value,
    sustainable_growth,
    variance_bound,
    wacc,
)
from financialkit.asset_pricing.utils.sorts import long_short_factor, momentum_signal, sort_portfolios
from financialkit.asset_pricing.utils.synthetic import FactorPanel, simulate_factor_model

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
    "LucasTree",
    "MehraPrescott",
    "ccapm_expected_excess_return",
    "implied_risk_aversion",
    "FiniteStateMarket",
    "hansen_jagannathan_bound",
    "max_sharpe_ratio",
    "sort_portfolios",
    "long_short_factor",
    "momentum_signal",
    "FactorPanel",
    "simulate_factor_model",
]
