"""Exotic option pricers: barriers, binaries, Asians, lookbacks, variance
swaps, compounds and choosers, multi-asset options and American options
by least-squares Monte Carlo.
"""

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
from financialkit.exotic_options.systems.variance_swaps import realized_variance, variance_swap_strike

__all__ = [
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
]
