"""Matplotlib visualizers for financialkit.fixed_income."""

from financialkit.fixed_income.visualizers.plots import (
    plot_cash_flows,
    plot_discounted_cash_flows,
    plot_key_rate_durations,
    plot_mbs_cash_flows,
    plot_pca_loadings,
    plot_price_yield,
    plot_surplus,
    plot_yield_curve,
)

__all__ = [
    "plot_yield_curve",
    "plot_price_yield",
    "plot_discounted_cash_flows",
    "plot_cash_flows",
    "plot_key_rate_durations",
    "plot_pca_loadings",
    "plot_surplus",
    "plot_mbs_cash_flows",
]
