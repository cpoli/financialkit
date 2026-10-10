"""financialkit.monte_carlo: Monte Carlo estimation, variance reduction, quasi-Monte Carlo and Greeks.

Crude Monte Carlo after Metropolis and Ulam, with standard errors and
confidence intervals on every estimate; Box-Muller normal variates; and
Boyle's option prices under Black-Scholes, simulated through
:mod:`financialkit.integrators`. Variance reduction: Neyman's stratified
sampling, importance sampling by a mean shift, antithetic variates,
Latin hypercube sampling (:mod:`scipy.stats.qmc`), regression control
variates and Barraquand's moment matching. Quasi-Monte Carlo: van der
Corput, Halton and Sobol sequences, exact and :math:`L^2` star
discrepancies with the Koksma-Hlawka bound, Owen-scrambled randomized
QMC with error bars, and high-dimensional path integrals with and
without Brownian-bridge ordering. Greeks by the likelihood-ratio and
pathwise methods, and Giles's adaptive multilevel Monte Carlo.
"""

from financialkit import __version__
from financialkit.monte_carlo.core.base import (
    ControlVariateResult,
    MLMCResult,
    MonteCarloGreeks,
    MonteCarloResult,
    StratifiedResult,
)
from financialkit.monte_carlo.systems.basics import (
    box_muller,
    box_muller_normals,
    european_option_mc,
    monte_carlo_integrate,
)
from financialkit.monte_carlo.systems.greeks import (
    likelihood_ratio_greeks,
    pathwise_greeks,
)
from financialkit.monte_carlo.systems.multilevel import (
    mlmc_level,
    multilevel_monte_carlo,
)
from financialkit.monte_carlo.systems.quasi_monte_carlo import (
    gbm_path_estimate,
    halton,
    koksma_hlawka_bound,
    l2_star_discrepancy,
    randomized_qmc_estimate,
    sobol,
    star_discrepancy,
    van_der_corput,
)
from financialkit.monte_carlo.systems.variance_reduction import (
    antithetic_estimate,
    control_variate_estimate,
    importance_sampling_normal,
    latin_hypercube_estimate,
    moment_match,
    moment_matching_estimate,
    neyman_allocation,
    stratified_estimate,
)
from financialkit.monte_carlo.utils.numerics import (
    sample_mean,
    terminal_prices,
    vanilla_payoff,
)
from financialkit.monte_carlo.visualizers.plots import (
    plot_allocation,
    plot_confidence_intervals,
    plot_error_convergence,
    plot_estimator_comparison,
    plot_mlmc,
    plot_point_set,
)

__all__ = [
    "__version__",
    "MonteCarloResult",
    "StratifiedResult",
    "ControlVariateResult",
    "MonteCarloGreeks",
    "MLMCResult",
    "monte_carlo_integrate",
    "box_muller",
    "box_muller_normals",
    "european_option_mc",
    "neyman_allocation",
    "stratified_estimate",
    "importance_sampling_normal",
    "antithetic_estimate",
    "latin_hypercube_estimate",
    "control_variate_estimate",
    "moment_match",
    "moment_matching_estimate",
    "van_der_corput",
    "halton",
    "sobol",
    "star_discrepancy",
    "l2_star_discrepancy",
    "koksma_hlawka_bound",
    "randomized_qmc_estimate",
    "gbm_path_estimate",
    "likelihood_ratio_greeks",
    "pathwise_greeks",
    "mlmc_level",
    "multilevel_monte_carlo",
    "sample_mean",
    "terminal_prices",
    "vanilla_payoff",
    "plot_error_convergence",
    "plot_confidence_intervals",
    "plot_estimator_comparison",
    "plot_point_set",
    "plot_allocation",
    "plot_mlmc",
]
