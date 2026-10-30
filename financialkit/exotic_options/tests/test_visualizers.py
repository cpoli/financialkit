"""Smoke tests for financialkit.exotic_options.visualizers."""

import matplotlib.axes
import numpy as np

from financialkit.exotic_options import barrier_price, longstaff_schwartz, variance_swap_strike
from financialkit.exotic_options.visualizers import plot_barrier_paths, plot_exercise_boundary, plot_price_profile, plot_variance_strip
from financialkit.integrators import gbm_paths
from financialkit.options import black_scholes_price


def test_plots():
    S = np.linspace(80.0, 120.0, 5)
    ax = plot_price_profile(
        S, {"barrier": barrier_price(S, 100.0, 85.0, 1.0, 0.05, 0.2), "vanilla": black_scholes_price(S, 100.0, 1.0, 0.05, 0.2)}, markers=(85.0,)
    )
    assert isinstance(ax, matplotlib.axes.Axes)
    t, paths = gbm_paths(100.0, 0.05, 0.3, 1.0, 20, 10, rng=0)
    assert isinstance(plot_barrier_paths(t, paths, [85.0, 120.0]), matplotlib.axes.Axes)
    assert isinstance(plot_exercise_boundary(longstaff_schwartz(36.0, 40.0, 1.0, 0.06, 0.2, n_steps=10, n_paths=2000, rng=0)), matplotlib.axes.Axes)
    K = np.arange(50.0, 200.0, 5.0)
    res = variance_swap_strike(
        K, black_scholes_price(100.0, K, 1.0, 0.0, 0.2), black_scholes_price(100.0, K, 1.0, 0.0, 0.2, option_type="put"), 100.0, 1.0, 0.0
    )
    assert isinstance(plot_variance_strip(res), matplotlib.axes.Axes)
