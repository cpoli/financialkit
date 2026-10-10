"""Test-session setup: force Matplotlib's non-interactive Agg backend
before anything imports pyplot, so the suite (and CI, which has no
display) can exercise the plotting code headlessly, and close every
figure after each test so a full-suite run doesn't accumulate open
figures across the many visualizer smoke tests."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_all_figures():
    yield
    plt.close("all")
