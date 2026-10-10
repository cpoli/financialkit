"""Smoke tests for financialkit.risk.visualizers."""

import matplotlib.axes
import numpy as np

from financialkit.risk import drawdowns, euler_allocation, fit_gev, fit_pot, gaussian_copula_sample, historical_var, rolling_historical_var
from financialkit.risk.visualizers import plot_allocation, plot_backtest, plot_copula, plot_drawdowns, plot_gev_fit, plot_pot_tail, plot_var

X = np.random.default_rng(0).standard_t(4, 2000) * 0.01


def test_plot_var_and_backtest():
    assert isinstance(plot_var(historical_var(X, 0.99)), matplotlib.axes.Axes)
    assert isinstance(plot_backtest(X[250:], rolling_historical_var(X, 250)), matplotlib.axes.Axes)


def test_plot_evt():
    assert isinstance(plot_gev_fit(fit_gev(X, 20)), matplotlib.axes.Axes)
    assert isinstance(plot_pot_tail(fit_pot(X, 0.02)), matplotlib.axes.Axes)


def test_plot_copula_allocation_drawdowns():
    assert isinstance(plot_copula(gaussian_copula_sample([[1, 0.5], [0.5, 1]], 200, 0), title="g"), matplotlib.axes.Axes)
    assert isinstance(plot_allocation(euler_allocation([1.0, 2.0], np.eye(2) * 1e-4)), matplotlib.axes.Axes)
    assert len(plot_drawdowns(drawdowns(X))) == 2
