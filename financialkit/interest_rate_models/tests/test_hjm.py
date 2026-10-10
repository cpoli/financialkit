"""Tests for the HJM drift condition and forward-curve simulation."""

import numpy as np
import pytest

from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import HoLeeModel, hjm_drift, simulate_hjm

CURVE = NelsonSiegelCurve(0.05, -0.02, 0.015, 2.0)


def test_hjm_drift_for_ho_lee_and_hull_white_volatilities():
    T = np.array([1.0, 5.0, 10.0])
    assert np.allclose(hjm_drift(lambda t, T: 0.01, 0.0, T), 0.01**2 * T)
    s, a, t = 0.01, 0.2, 1.0
    vol = lambda t, T: s * np.exp(-a * (T - t))
    tau = T - t
    expected = s**2 * np.exp(-a * tau) * (1 - np.exp(-a * tau)) / a
    assert np.allclose(hjm_drift(vol, t, T), expected, rtol=1e-10)


def test_drift_condition_reprices_the_initial_curve():
    res = simulate_hjm(CURVE, 0.02, 0.1, horizon=10.0, n_steps=50, n_paths=4000, rng=1)
    prices, se = res.bond_prices()
    exact = CURVE.discount(res.times)
    assert prices[0] == 1.0
    assert np.all(np.abs(prices[1:] - exact[1:]) < 4 * se[1:])


def test_without_the_drift_condition_bonds_are_mispriced():
    res = simulate_hjm(CURVE, 0.02, 0.0, horizon=10.0, n_steps=50, n_paths=4000, drift_condition=False, rng=1)
    prices, se = res.bond_prices()
    gap = prices[-1] - CURVE.discount(10.0)
    assert gap > 10 * se[-1]
    assert not res.drift_condition


def test_constant_volatility_hjm_short_rate_is_ho_lee():
    """sigma(t, T) = sigma: the short rate has Ho-Lee's mean f(0, t) + sigma^2 t^2 / 2 and variance sigma^2 t."""
    res = simulate_hjm(CURVE, 0.01, 0.0, horizon=4.0, n_steps=40, n_paths=8000, rng=2)
    r = res.short_rates[:, -1]
    mean = float(HoLeeModel(CURVE, 0.01).shift(4.0))
    assert abs(r.mean() - mean) < 4 * r.std(ddof=1) / np.sqrt(r.size) + 0.01**2 * 4.0 * 0.1  # + Euler drift bias, sigma^2 t dt
    assert r.var() == pytest.approx(0.01**2 * 4.0, rel=0.05)
    assert res.forwards.shape == (8000, 41, 41)
