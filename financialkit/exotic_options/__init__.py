"""financialkit.exotic_options: barrier, binary, Asian, lookback, compound, multi-asset and American options.

Barriers: Merton's down-and-out call by the method of images, the
Reiner-Rubinstein closed forms for all eight single-barrier options with
rebates, Ikeda-Kunitomo double barriers, the Broadie-Glasserman-Kou
correction for discrete monitoring, and Monte Carlo with Brownian-bridge
crossing probabilities. Binaries: cash-or-nothing, asset-or-nothing, gap
and one-touch options. Asians: Kemna-Vorst geometric closed forms, the
arithmetic average by Monte Carlo with the geometric control variate,
and Turnbull-Wakeman moment matching. Lookbacks: Goldman-Sosin-Gatto
floating and Conze-Viswanathan fixed strikes, with exact bridge-sampled
extrema for Monte Carlo. Variance swaps replicated by a strip of
out-of-the-money options (Demeterfi-Derman-Kamal-Zou). Geske compound
options, Rubinstein forward starts and choosers, Margrabe exchange
options, Stulz options on the maximum or minimum of two assets, Reiner
quantos, and American options by Longstaff-Schwartz least-squares Monte
Carlo. Paths are simulated through :mod:`financialkit.integrators`, and
Monte Carlo results are :mod:`financialkit.monte_carlo` result objects.
"""

from financialkit import __version__
from financialkit.exotic_options.core.base import LongstaffSchwartzResult, VarianceSwapResult
from financialkit.exotic_options.systems.american import longstaff_schwartz, regression_basis
from financialkit.exotic_options.systems.asian import arithmetic_asian_mc, geometric_asian_price, turnbull_wakeman_price
from financialkit.exotic_options.systems.barriers import (
    BGK_BETA,
    barrier_option_mc,
    barrier_price,
    discrete_barrier_price,
    double_barrier_price,
    merton_down_and_out_call,
)
from financialkit.exotic_options.systems.binaries import asset_or_nothing_price, cash_or_nothing_price, gap_option_price, one_touch_price
from financialkit.exotic_options.systems.compound import chooser_price, compound_option_price, critical_price, forward_start_price
from financialkit.exotic_options.systems.lookbacks import fixed_lookback_price, floating_lookback_price, lookback_option_mc
from financialkit.exotic_options.systems.multi_asset import margrabe_price, quanto_price, stulz_price
from financialkit.exotic_options.systems.variance_swaps import annualized_realized_variance, realized_variance, variance_swap_strike
from financialkit.exotic_options.utils.numerics import (
    barrier_crossing_probability,
    bivariate_normal_cdf,
    bridge_extremum,
    parse_barrier_type,
)
from financialkit.exotic_options.visualizers.plots import (
    plot_barrier_paths,
    plot_exercise_boundary,
    plot_price_profile,
    plot_variance_strip,
)

__all__ = [
    "__version__",
    "LongstaffSchwartzResult",
    "VarianceSwapResult",
    "BGK_BETA",
    "merton_down_and_out_call",
    "barrier_price",
    "double_barrier_price",
    "discrete_barrier_price",
    "barrier_option_mc",
    "cash_or_nothing_price",
    "asset_or_nothing_price",
    "gap_option_price",
    "one_touch_price",
    "geometric_asian_price",
    "arithmetic_asian_mc",
    "turnbull_wakeman_price",
    "floating_lookback_price",
    "fixed_lookback_price",
    "lookback_option_mc",
    "variance_swap_strike",
    "annualized_realized_variance",
    "realized_variance",
    "compound_option_price",
    "critical_price",
    "forward_start_price",
    "chooser_price",
    "margrabe_price",
    "stulz_price",
    "quanto_price",
    "longstaff_schwartz",
    "regression_basis",
    "bivariate_normal_cdf",
    "parse_barrier_type",
    "barrier_crossing_probability",
    "bridge_extremum",
    "plot_price_profile",
    "plot_barrier_paths",
    "plot_exercise_boundary",
    "plot_variance_strip",
]
