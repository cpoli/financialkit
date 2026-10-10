"""Smoke tests for financialkit.portfolio.visualizers."""

import matplotlib.axes
import numpy as np

from financialkit.integrators import gbm_paths
from financialkit.portfolio import cppi, efficient_frontier, equal_weight, out_of_sample_returns
from financialkit.portfolio.visualizers import plot_cppi, plot_frontier, plot_out_of_sample, plot_weights

MU = np.array([0.05, 0.08, 0.12])
COV = np.diag([0.01, 0.03, 0.06])


def test_plot_frontier_and_weights():
    fr = efficient_frontier(MU, COV, 8, long_only=True)
    assert isinstance(plot_frontier(fr, labels=["a", "b", "c"], rf=0.02), matplotlib.axes.Axes)
    assert isinstance(plot_weights(fr.weights, x=fr.volatilities), matplotlib.axes.Axes)
    assert isinstance(plot_weights(fr.weights[0], labels=["a", "b", "c"]), matplotlib.axes.Axes)


def test_plot_cppi_and_out_of_sample():
    times, S = gbm_paths(100.0, 0.05, 0.3, 1.0, 12, 10, rng=0)
    assert isinstance(plot_cppi(cppi(S, times)), matplotlib.axes.Axes)
    R = np.random.default_rng(0).normal(0.01, 0.05, (30, 3))
    assert isinstance(plot_out_of_sample([out_of_sample_returns(R, lambda p: equal_weight(3), 10, "1/N")]), matplotlib.axes.Axes)
