"""Smoke tests for financialkit.monte_carlo.visualizers."""

import matplotlib.axes
import numpy as np

from financialkit.monte_carlo import monte_carlo_integrate, multilevel_monte_carlo, sobol, stratified_estimate
from financialkit.monte_carlo.visualizers import (
    plot_allocation,
    plot_confidence_intervals,
    plot_error_convergence,
    plot_estimator_comparison,
    plot_mlmc,
    plot_point_set,
)


def test_plots():
    res = [monte_carlo_integrate(lambda u: u[:, 0], 100, rng=s) for s in range(5)]
    assert isinstance(plot_error_convergence([10, 100, 1000], [0.1, 0.03, 0.01], rates=(0.5, 1.0)), matplotlib.axes.Axes)
    assert isinstance(plot_confidence_intervals(res, exact=0.5), matplotlib.axes.Axes)
    assert isinstance(plot_estimator_comparison(res[:3], exact=0.5), matplotlib.axes.Axes)
    assert isinstance(plot_point_set(sobol(64, 2, rng=0), grid=8, title="Sobol"), matplotlib.axes.Axes)
    assert isinstance(plot_allocation(stratified_estimate(lambda u: u**2, 200, n_strata=4, rng=0)), matplotlib.axes.Axes)
    axes = plot_mlmc(multilevel_monte_carlo(100.0, 100.0, 1.0, 0.05, 0.2, eps=0.5, rng=0))
    assert isinstance(axes, np.ndarray) and axes.size == 2
