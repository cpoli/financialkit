"""Smoke tests for financialkit.asset_pricing.visualizers."""

import matplotlib.axes
import numpy as np

from financialkit.asset_pricing import (
    LucasTree,
    ar1_dividend_price,
    capm_equilibrium,
    fama_macbeth,
    hansen_jagannathan_bound,
    simulate_factor_model,
    time_series_regression,
    variance_bound,
)
from financialkit.asset_pricing.visualizers import plot_alphas, plot_excess_volatility, plot_fama_macbeth, plot_hj_bound, plot_sml

PANEL = simulate_factor_model(120, np.linspace(0.5, 1.5, 8), [0.006], [[0.002]], rng=0)


def test_plot_sml_and_alphas():
    eq = capm_equilibrium([0.05, 0.08], [[0.01, 0.002], [0.002, 0.04]], 0.02)
    assert isinstance(plot_sml(eq, labels=["a", "b"]), matplotlib.axes.Axes)
    assert isinstance(plot_alphas(time_series_regression(PANEL.excess_returns, PANEL.factors)), matplotlib.axes.Axes)


def test_plot_fama_macbeth_and_shiller():
    assert isinstance(plot_fama_macbeth(fama_macbeth(PANEL.excess_returns, factors=PANEL.factors), true_value=0.006), matplotlib.axes.Axes)
    D = 1 + 0.1 * np.random.default_rng(0).standard_normal(50)
    assert isinstance(plot_excess_volatility(variance_bound(ar1_dividend_price(D, 0.05, 0.0, 1.0), D, 0.05)), matplotlib.axes.Axes)


def test_plot_hj_bound():
    b = hansen_jagannathan_bound([1.07], [[0.03]], np.linspace(0.9, 1.0, 20))
    assert isinstance(plot_hj_bound(b, {"CRRA": LucasTree(0.99, 5.0).moments()}), matplotlib.axes.Axes)
