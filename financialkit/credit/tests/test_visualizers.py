"""Smoke tests for financialkit.credit.visualizers."""

import matplotlib.axes
import numpy as np

from financialkit.credit import (
    HazardCurve,
    creditmetrics_bbb_values,
    creditmetrics_bond,
    creditmetrics_matrix,
    creditmetrics_portfolio,
    creditrisk_plus,
    tranche_spread,
)
from financialkit.credit.visualizers import plot_loss_distribution, plot_survival, plot_tranche_losses, plot_transition_matrix, plot_value_distribution

P = creditmetrics_matrix()


def test_plot_survival_and_losses():
    assert isinstance(plot_survival(HazardCurve([1, 5], [0.01, 0.03]), label="x"), matplotlib.axes.Axes)
    assert isinstance(plot_loss_distribution(creditrisk_plus(np.ones(20), np.full(20, 0.05), 0.5), log=True), matplotlib.axes.Axes)


def test_plot_matrices_and_values():
    assert isinstance(plot_transition_matrix(P, labels=list("ABCDEFGH")), matplotlib.axes.Axes)
    assert isinstance(plot_value_distribution(creditmetrics_bond(P[3], creditmetrics_bbb_values())), matplotlib.axes.Axes)
    assert isinstance(plot_value_distribution(creditmetrics_portfolio(P, [3, 3], creditmetrics_bbb_values(), 0.3, 500, rng=0)), matplotlib.axes.Axes)


def test_plot_tranches():
    res = [tranche_spread(a, d, HazardCurve.flat(0.01), 0.3, 3.0, 0.03) for a, d in ((0.0, 0.03), (0.03, 0.07))]
    assert isinstance(plot_tranche_losses(res), matplotlib.axes.Axes)
