"""financialkit.risk: value at risk, expected shortfall, extreme values, backtests, copulas and drawdowns.

Value at risk by historical simulation, normal and Student-t closed
forms, RiskMetrics delta-normal, the Cornish-Fisher expansion, full
revaluation Monte Carlo (through :mod:`financialkit.integrators`),
Boudoukh-Richardson-Whitelaw age weighting, filtered historical
simulation, Rouvinez's exact delta-gamma VaR and
Bangia-Diebold-Schuermann-Stroughair liquidity-adjusted VaR; Roy's
safety-first criterion; Acerbi-Tasche expected shortfall, the coherence
axioms, Euler risk allocation and the FRTB's liquidity-horizon ES;
block maxima and peaks over threshold (:data:`scipy.stats.genextreme`,
:data:`scipy.stats.genpareto`); Kupiec and Christoffersen backtests,
the Basel traffic light and Acerbi-Szekely ES backtests; Gaussian and t copulas with tail dependence;
drawdowns and the Calmar ratio. Profits and losses go in with losses
negative, and every risk figure comes out as a positive loss.
"""

from financialkit import __version__
from financialkit.risk.core.base import (
    AllocationResult,
    BacktestResult,
    CoherenceResult,
    DrawdownResult,
    ESBacktestResult,
    FRTBResult,
    GEVResult,
    GPDResult,
    RiskMeasure,
    TrafficLightResult,
    VaRResult,
)
from financialkit.risk.systems.allocation import euler_allocation, scenario_es_allocation
from financialkit.risk.systems.backtesting import (
    acerbi_szekely_test,
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    basel_traffic_light,
    christoffersen_independence,
    christoffersen_test,
    kupiec_test,
)
from financialkit.risk.systems.coherence import ExpectedShortfall, ValueAtRisk, check_coherence
from financialkit.risk.systems.copulas import (
    apply_marginals,
    copula_kendall_tau,
    empirical_tail_dependence,
    gaussian_copula_sample,
    t_copula_sample,
    t_tail_dependence,
)
from financialkit.risk.systems.delta_gamma import delta_gamma_cdf, delta_gamma_cumulants, delta_gamma_var
from financialkit.risk.systems.drawdowns import calmar_ratio, drawdowns
from financialkit.risk.systems.expected_shortfall import expected_shortfall, parametric_es
from financialkit.risk.systems.extreme_value import block_maxima, fit_gev, fit_pot, gev_return_level, mean_excess, pot_es, pot_var
from financialkit.risk.systems.frtb import LIQUIDITY_HORIZONS, frtb_es_cascade, frtb_expected_shortfall
from financialkit.risk.systems.liquidity import exogenous_liquidity_cost, liquidity_adjusted_var
from financialkit.risk.systems.safety_first import safety_first_portfolio, safety_first_ratio, shortfall_probability
from financialkit.risk.systems.var import (
    cornish_fisher_var,
    delta_normal_var,
    filtered_historical_var,
    historical_var,
    hybrid_var,
    monte_carlo_var,
    parametric_var,
    rolling_historical_var,
)
from financialkit.risk.utils.tail import pseudo_observations, var_exceptions, weighted_tail
from financialkit.risk.visualizers.plots import (
    plot_allocation,
    plot_backtest,
    plot_copula,
    plot_drawdowns,
    plot_frtb_cascade,
    plot_gev_fit,
    plot_pot_tail,
    plot_var,
)

__all__ = [
    "__version__",
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
    "historical_var",
    "rolling_historical_var",
    "parametric_var",
    "delta_normal_var",
    "cornish_fisher_var",
    "monte_carlo_var",
    "hybrid_var",
    "filtered_historical_var",
    "delta_gamma_cumulants",
    "delta_gamma_cdf",
    "delta_gamma_var",
    "exogenous_liquidity_cost",
    "liquidity_adjusted_var",
    "safety_first_ratio",
    "shortfall_probability",
    "safety_first_portfolio",
    "expected_shortfall",
    "parametric_es",
    "LIQUIDITY_HORIZONS",
    "frtb_es_cascade",
    "frtb_expected_shortfall",
    "ValueAtRisk",
    "ExpectedShortfall",
    "check_coherence",
    "euler_allocation",
    "scenario_es_allocation",
    "block_maxima",
    "fit_gev",
    "gev_return_level",
    "mean_excess",
    "fit_pot",
    "pot_var",
    "pot_es",
    "kupiec_test",
    "christoffersen_independence",
    "christoffersen_test",
    "basel_traffic_light",
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "acerbi_szekely_test",
    "gaussian_copula_sample",
    "t_copula_sample",
    "apply_marginals",
    "t_tail_dependence",
    "empirical_tail_dependence",
    "copula_kendall_tau",
    "drawdowns",
    "calmar_ratio",
    "weighted_tail",
    "var_exceptions",
    "pseudo_observations",
    "plot_var",
    "plot_backtest",
    "plot_gev_fit",
    "plot_pot_tail",
    "plot_copula",
    "plot_allocation",
    "plot_drawdowns",
    "plot_frtb_cascade",
]
