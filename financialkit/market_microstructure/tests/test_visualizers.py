"""Smoke tests for financialkit.market_microstructure.visualizers."""

import matplotlib.axes
import numpy as np

from financialkit.market_microstructure import (
    AvellanedaStoikov,
    almgren_chriss,
    hasbrouck_var,
    simulate_garman,
    simulate_glosten_milgrom,
    simulate_market_making,
    vpin,
)
from financialkit.market_microstructure.visualizers import (
    plot_execution,
    plot_garman,
    plot_glosten_milgrom,
    plot_impulse_response,
    plot_market_making,
    plot_quotes,
    plot_vpin,
)

RNG = np.random.default_rng(0)


def test_plot_quotes_and_glosten_milgrom():
    p = 10 + RNG.choice([-0.05, 0.05], 20)
    assert isinstance(plot_quotes(p, np.full(20, 9.95), np.full(20, 10.05)), matplotlib.axes.Axes)
    assert isinstance(plot_quotes(p, np.full(20, 9.95), np.full(20, 10.05), side=np.sign(p - 10)), matplotlib.axes.Axes)
    assert isinstance(plot_glosten_milgrom(simulate_glosten_milgrom(30, 0.3, 0.0, 1.0, rng=0)), matplotlib.axes.Axes)


def test_plot_dealers():
    assert isinstance(plot_garman(simulate_garman(3, 5.0, 1.0, 1.0, 1.0, 1.0, 50, 5, rng=0)), matplotlib.axes.Axes)
    mm = simulate_market_making(AvellanedaStoikov(0.1, 2.0, 1.5, 1.0), 100.0, 2.0, 1.0, 20, 140.0, 1.5, 2, rng=0)
    assert len(plot_market_making(mm)) == 2


def test_plot_execution_impulse_vpin():
    res = almgren_chriss(100.0, 1.0, 5, 0.3, 0.1, 1.0)
    assert isinstance(plot_execution(res), matplotlib.axes.Axes)
    assert isinstance(plot_execution([res, res], labels=["a", "b"]), matplotlib.axes.Axes)
    x = RNG.choice([-1.0, 1.0], 500)
    assert isinstance(plot_impulse_response(hasbrouck_var(0.1 * x + 0.05 * RNG.standard_normal(500), x, lags=2)), matplotlib.axes.Axes)
    assert isinstance(plot_vpin(vpin(RNG.standard_normal(200), np.full(200, 10.0), 100.0, window=5)), matplotlib.axes.Axes)
