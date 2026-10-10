"""Smoke tests for financialkit.options.visualizers."""

import matplotlib.axes
import numpy as np
import pytest

from financialkit.options import (
    CRRTree,
    black_scholes_greeks,
    black_scholes_price,
    breeden_litzenberger_density,
    delta_hedge,
    implied_vol_smile,
)
from financialkit.options.visualizers import (
    plot_density,
    plot_exercise_boundary,
    plot_greeks,
    plot_hedge_pnl,
    plot_option_value,
    plot_smile,
    plot_tree_convergence,
)

S = np.linspace(60, 140, 41)


def test_plot_option_value():
    assert isinstance(plot_option_value(S, black_scholes_price(S, 100.0, 1.0, 0.05, 0.2), 100.0), matplotlib.axes.Axes)


def test_plot_greeks():
    axes = plot_greeks(S, black_scholes_greeks(S, 100.0, 1.0, 0.05, 0.2), names=("delta", "gamma", "vega", "theta"))
    assert len(axes) == 6 and not axes[-1].get_visible()


def test_plot_tree_convergence():
    n = np.array([10, 20, 40])
    prices = [CRRTree(k).price(100.0, 100.0, 1.0, 0.05, 0.2).price for k in n]
    assert isinstance(plot_tree_convergence(n, prices, black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)), matplotlib.axes.Axes)


def test_plot_exercise_boundary():
    res = CRRTree(50).price(100.0, 100.0, 1.0, 0.08, 0.25, option_type="put", american=True)
    assert isinstance(plot_exercise_boundary(res), matplotlib.axes.Axes)
    with pytest.raises(ValueError):
        plot_exercise_boundary(CRRTree(50).price(100.0, 100.0, 1.0, 0.08, 0.25))


def test_plot_smile_and_density():
    K = np.linspace(80, 120, 21)
    c = black_scholes_price(100.0, K, 1.0, 0.0, 0.2)
    assert isinstance(plot_smile(implied_vol_smile(c, K, 100.0, 1.0, 0.0), moneyness=True), matplotlib.axes.Axes)
    assert isinstance(plot_density(breeden_litzenberger_density(K, c, 1.0, 0.0)), matplotlib.axes.Axes)


def test_plot_hedge_pnl():
    assert isinstance(plot_hedge_pnl(delta_hedge(100.0, 100.0, 1.0, 0.05, 0.2, 10, 200, rng=0)), matplotlib.axes.Axes)
