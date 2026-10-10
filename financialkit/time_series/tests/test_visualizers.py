"""Smoke tests for financialkit.time_series.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np
import pytest

from financialkit.time_series import EGARCH, GARCH, GJRGARCH, hill_estimator, hurst_exponent, variance_ratio
from financialkit.time_series.visualizers import (
    plot_acf,
    plot_conditional_volatility,
    plot_hill,
    plot_news_impact,
    plot_rescaled_range,
    plot_tail,
    plot_variance_ratio,
)


def test_plots_return_axes():
    rng = np.random.default_rng(0)
    x = rng.standard_t(4, 2000)
    path = GARCH(0.05, 0.1, 0.85).simulate(500, rng=1)
    axes = [
        plot_acf(x, 10, label="t returns"),
        plot_acf(x, 10, partial=True),
        plot_tail(x, label="t(4)"),
        plot_hill(hill_estimator(np.abs(x)), true_alpha=4.0, label="Hill"),
        plot_rescaled_range(hurst_exponent(x)),
        plot_variance_ratio(variance_ratio(np.cumsum(x), q=[2, 4, 8]), label="VR"),
        plot_conditional_volatility(path.returns, path.variance, periods_per_year=252),
        plot_news_impact({"GARCH": GARCH(0.05, 0.1, 0.85), "GJR": GJRGARCH(0.05, 0.02, 0.15, 0.85), "EGARCH": EGARCH(0.0, 0.2, -0.1, 0.95)}, var_prev=1.0),
    ]
    assert all(isinstance(ax, matplotlib.axes.Axes) for ax in axes)
    with pytest.raises(TypeError):
        plot_news_impact({"bad": object()}, var_prev=1.0)
