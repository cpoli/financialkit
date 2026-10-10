"""Risk measures and models: VaR by every method, expected shortfall and
coherence, Euler allocation, extreme value theory, backtests, copulas
and drawdowns.
"""

from financialkit.risk.systems.allocation import euler_allocation, scenario_es_allocation
from financialkit.risk.systems.backtesting import basel_traffic_light, christoffersen_independence, christoffersen_test, kupiec_test
from financialkit.risk.systems.coherence import ExpectedShortfall, ValueAtRisk, check_coherence
from financialkit.risk.systems.copulas import (
    apply_marginals,
    copula_kendall_tau,
    empirical_tail_dependence,
    gaussian_copula_sample,
    t_copula_sample,
    t_tail_dependence,
)
from financialkit.risk.systems.drawdowns import calmar_ratio, drawdowns
from financialkit.risk.systems.expected_shortfall import expected_shortfall, parametric_es
from financialkit.risk.systems.extreme_value import block_maxima, fit_gev, fit_pot, gev_return_level, mean_excess, pot_es, pot_var
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

__all__ = [
    "historical_var",
    "rolling_historical_var",
    "parametric_var",
    "delta_normal_var",
    "cornish_fisher_var",
    "monte_carlo_var",
    "hybrid_var",
    "filtered_historical_var",
    "safety_first_ratio",
    "shortfall_probability",
    "safety_first_portfolio",
    "expected_shortfall",
    "parametric_es",
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
    "gaussian_copula_sample",
    "t_copula_sample",
    "apply_marginals",
    "t_tail_dependence",
    "empirical_tail_dependence",
    "copula_kendall_tau",
    "drawdowns",
    "calmar_ratio",
]
