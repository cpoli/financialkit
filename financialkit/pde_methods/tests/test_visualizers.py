"""Smoke tests for financialkit.pde_methods.visualizers."""

import matplotlib.axes
import numpy as np
import pytest

from financialkit.pde_methods import CrankNicolson, HestonADI, PeacemanRachfordADI, heat_equation_price, tavella_randall_grid, von_neumann_amplification
from financialkit.pde_methods.visualizers import (
    plot_adi_surface,
    plot_amplification,
    plot_exercise_boundary,
    plot_grid,
    plot_grid_convergence,
    plot_heat_solution,
    plot_heston_surface,
    plot_value_profile,
    plot_value_surface,
)


def test_plots():
    eu = CrankNicolson(50, 20).price(100.0, 100.0, 1.0, 0.05, 0.2)
    am = CrankNicolson(50, 20).price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="put", american=True)
    assert isinstance(plot_value_profile(eu, exact=eu.V, xlim=(50, 150)), matplotlib.axes.Axes)
    assert isinstance(plot_value_surface(eu, s_range=(50, 150)), matplotlib.axes.Axes)
    assert isinstance(plot_exercise_boundary(am), matplotlib.axes.Axes)
    with pytest.raises(ValueError):
        plot_exercise_boundary(eu)
    vn = [von_neumann_amplification(0.2, 0.05, 0.01, 0.02, theta=t) for t in (0.0, 0.5)]
    assert isinstance(plot_amplification(vn), matplotlib.axes.Axes)
    assert isinstance(plot_amplification(vn[0], labels=["explicit"]), matplotlib.axes.Axes)
    assert isinstance(plot_grid_convergence([10, 20, 40], [1e-2, 2.5e-3, 6e-4]), matplotlib.axes.Axes)
    assert isinstance(plot_grid(tavella_randall_grid(0, 400, 40, 100, 10), K=100.0, label="sinh"), matplotlib.axes.Axes)
    assert isinstance(plot_heat_solution(heat_equation_price(100.0, 100.0, 1.0, 0.05, 0.2, n_space=50, n_time=10)), matplotlib.axes.Axes)
    adi = PeacemanRachfordADI(20, 20, 5).price(100.0, 100.0, 1.0, 0.05, 0.2, 0.3, 0.5, lambda a, b: np.maximum(a - b, 0.0))
    assert isinstance(plot_adi_surface(adi, log=True), matplotlib.axes.Axes)


def test_plot_heston_surface():
    res = HestonADI(20, 10, 5).price(100.0, 100.0, 1.0, 0.0, 0.04, 1.5, 0.04, 0.3, -0.5)
    assert isinstance(plot_heston_surface(res, s_range=(50, 150)), matplotlib.axes.Axes)
    assert isinstance(plot_heston_surface(res, v_max=None), matplotlib.axes.Axes)
