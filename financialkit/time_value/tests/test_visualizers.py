"""Smoke tests for financialkit.time_value.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from financialkit.time_value import amortization_schedule, npv_profile
from financialkit.time_value.visualizers import plot_amortization, plot_cash_flows, plot_compounding_growth, plot_npv_profile


def test_plot_cash_flows_returns_axes():
    assert isinstance(plot_cash_flows([-100.0, 30.0, 40.0, 50.0]), matplotlib.axes.Axes)


def test_plot_compounding_growth_returns_axes():
    assert isinstance(plot_compounding_growth(0.5, 3.0), matplotlib.axes.Axes)


def test_plot_npv_profile_returns_axes():
    profile = npv_profile([-1600.0, 10_000.0, -10_000.0], np.linspace(0.0, 5.0, 50))
    assert isinstance(plot_npv_profile(profile), matplotlib.axes.Axes)


def test_plot_amortization_returns_axes():
    assert isinstance(plot_amortization(amortization_schedule(1000.0, 0.12, 12)), matplotlib.axes.Axes)
