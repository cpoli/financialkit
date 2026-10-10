"""Smoke tests for financialkit.stochastic_volatility.visualizers."""

import matplotlib.axes
import numpy as np

from financialkit.options import black_scholes_price
from financialkit.stochastic_volatility import BlackScholesModel, HestonModel, cos_density, dupire_local_vol, heston_paths
from financialkit.stochastic_volatility.visualizers import plot_density, plot_heston_paths, plot_local_vol, plot_smile


def test_plot_smile():
    K = np.linspace(80, 120, 9)
    vols = HestonModel(0.04, 1.5, 0.04, 0.5, -0.7).implied_vol(100.0, K, 1.0, 0.0)
    assert isinstance(plot_smile(K, vols), matplotlib.axes.Axes)
    assert isinstance(plot_smile(K, vols, forward=100.0), matplotlib.axes.Axes)


def test_plot_local_vol():
    K, T = np.linspace(70, 130, 31), np.linspace(0.5, 1.0, 6)
    lv = dupire_local_vol(K, T, black_scholes_price(100.0, K[None, :], T[:, None], 0.0, 0.2), 0.0)
    assert isinstance(plot_local_vol(lv), matplotlib.axes.Axes)
    assert isinstance(plot_local_vol(lv, maturities=[0.6]), matplotlib.axes.Axes)


def test_plot_heston_paths():
    paths = heston_paths(HestonModel(0.04, 1.5, 0.04, 0.5, -0.7), 100.0, 1.0, 0.0, n_steps=10, n_paths=30, rng=0)
    assert len(plot_heston_paths(paths, n_show=5)) == 2


def test_plot_density():
    x = np.linspace(-0.5, 0.5, 51)
    assert isinstance(plot_density(x, cos_density(BlackScholesModel(0.2), x, 1.0), log=True), matplotlib.axes.Axes)
