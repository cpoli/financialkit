"""Smoke tests for financialkit.interest_rate_models.visualizers."""

import matplotlib.axes

from financialkit.interest_rate_models import (
    HullWhiteModel,
    VasicekModel,
    black_derman_toy_tree,
    cap_floor_price,
    jamshidian_swaption,
    monte_carlo_bond_prices,
    simulate_hjm,
)
from financialkit.interest_rate_models.visualizers import (
    plot_bond_monte_carlo,
    plot_caplets,
    plot_forward_curves,
    plot_jamshidian,
    plot_model_yields,
    plot_rate_tree,
    plot_short_rate_paths,
)

MODEL = VasicekModel(0.03, 0.5, 0.05, 0.01)


def test_plot_short_rate_paths_and_yields():
    times, r = MODEL.simulate(2.0, 20, 50, rng=0)
    assert isinstance(plot_short_rate_paths(times, r, n_show=5), matplotlib.axes.Axes)
    assert isinstance(plot_model_yields(MODEL, label="Vasicek"), matplotlib.axes.Axes)


def test_plot_bond_monte_carlo():
    assert isinstance(plot_bond_monte_carlo(monte_carlo_bond_prices(MODEL, [1.0, 2.0], n_steps=20, n_paths=200, rng=0)), matplotlib.axes.Axes)


def test_plot_rate_tree_caplets_jamshidian_and_hjm():
    assert isinstance(plot_rate_tree(black_derman_toy_tree(0.04, 0.2, 2.0, 8)), matplotlib.axes.Axes)
    assert isinstance(plot_caplets(cap_floor_price(0.04, 0.04, 2.0, 0.2)), matplotlib.axes.Axes)
    assert isinstance(plot_jamshidian(jamshidian_swaption(HullWhiteModel(0.04, 0.1, 0.01), 1.0, 2.0, 0.04)), matplotlib.axes.Axes)
    assert isinstance(plot_forward_curves(simulate_hjm(0.04, 0.01, horizon=2.0, n_steps=10, n_paths=10, rng=0), every=2), matplotlib.axes.Axes)
