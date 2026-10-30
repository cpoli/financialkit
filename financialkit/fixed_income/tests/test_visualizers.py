"""Smoke tests for financialkit.fixed_income.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from financialkit.fixed_income import (
    CashFlows,
    NelsonSiegelCurve,
    bond_cash_flows,
    key_rate_durations,
    mbs_cash_flows,
    redington_immunize,
    simulate_yield_changes,
    yield_curve_pca,
)
from financialkit.fixed_income.visualizers import (
    plot_discounted_cash_flows,
    plot_key_rate_durations,
    plot_mbs_cash_flows,
    plot_pca_loadings,
    plot_price_yield,
    plot_surplus,
    plot_yield_curve,
)


def test_plots_return_axes():
    curve = NelsonSiegelCurve(0.04, -0.02, 0.01, 2.0)
    cf = bond_cash_flows(100.0, 0.05, 10.0)
    tenors = np.array([1.0, 2, 5, 10, 30])
    imm = redington_immunize(CashFlows([7.0], [100.0]), [bond_cash_flows(1.0, 0.0, 2.0), bond_cash_flows(1.0, 0.0, 15.0)], curve)
    axes = [
        plot_yield_curve(curve, label="NS"),
        plot_price_yield(cf, 0.05, compounding=2),
        plot_discounted_cash_flows(cf, curve),
        plot_key_rate_durations(key_rate_durations(cf, curve, [2.0, 5.0, 10.0])),
        plot_pca_loadings(yield_curve_pca(simulate_yield_changes(tenors, 200, rng=0), tenors)),
        plot_surplus(imm),
        plot_mbs_cash_flows(mbs_cash_flows(100.0, 0.06, 360, psa=150), label="150 PSA"),
    ]
    assert all(isinstance(ax, matplotlib.axes.Axes) for ax in axes)
