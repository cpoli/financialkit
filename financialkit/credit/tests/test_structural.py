"""Tests for Merton's model and its calibration, KMV distance to default, Black-Cox and Leland."""

import numpy as np
import pytest
from scipy import stats

from financialkit.constants import DEFAULT_RTOL
from financialkit.credit import (
    black_cox_default_probability,
    black_cox_survival,
    distance_to_default,
    first_passage_crossing_probability,
    kmv_default_point,
    kmv_iterate,
    leland_model,
    leland_optimal_coupon,
    merton_calibrate,
    merton_model,
)
from financialkit.integrators import gbm_paths
from financialkit.options import black_scholes_price


def test_merton_hull_example_24_3():
    """Hull, *Options, Futures, and Other Derivatives*, 10th ed., Example 24.3."""
    m = merton_calibrate(3.0, 0.80, 10.0, 1.0, 0.05)
    assert m.asset_value == pytest.approx(12.40, abs=0.005)
    assert m.asset_volatility == pytest.approx(0.2123, abs=5e-5)
    assert m.default_probability == pytest.approx(0.127, abs=5e-4)
    assert m.debt == pytest.approx(9.40, abs=0.005)
    assert m.equity == pytest.approx(3.0, rel=1e-9) and m.equity_volatility == pytest.approx(0.80, rel=1e-8)


def test_merton_debt_is_riskless_bond_minus_put():
    V, D, T, r, s = 120.0, 100.0, 3.0, 0.04, 0.3
    m = merton_model(V, D, T, r, s)
    put = black_scholes_price(V, D, T, r, s, option_type="put")
    assert m.debt == pytest.approx(D * np.exp(-r * T) - put, rel=DEFAULT_RTOL)
    assert m.distance_to_default == pytest.approx(float(distance_to_default(V, D, r, s, T)), rel=DEFAULT_RTOL)
    # Expected loss = PD * (1 - recovery) * D, discounted, equals the put value.
    assert np.exp(-r * T) * D * m.default_probability * (1 - m.recovery_rate) == pytest.approx(put, rel=1e-10)
    assert m.credit_spread > 0


def test_kmv_default_point():
    assert kmv_default_point(40.0, 60.0) == 70.0


def test_kmv_iteration_recovers_asset_path():
    rng = np.random.default_rng(1)
    r, D = 0.03, 70.0
    _, V = gbm_paths(100.0, 0.08, 0.25, T=2.0, n_steps=504, n_paths=1, rng=rng)
    V = V[0]
    # Price equity at the path's own sample volatility, so (V, s) is the iteration's fixed point.
    s = np.std(np.diff(np.log(V)), ddof=1) * np.sqrt(252)
    E = np.asarray(black_scholes_price(V, D, 1.0, r, s))
    res = kmv_iterate(E, D, r)
    # Both agree up to the stopping tolerance on sigma (DEFAULT_RTOL), amplified by the iteration.
    assert res.asset_volatility == pytest.approx(s, rel=1e3 * DEFAULT_RTOL)
    np.testing.assert_allclose(res.asset_values, V, rtol=1e3 * DEFAULT_RTOL)
    # The naive start (equity vol scaled by E / (E + D)) is biased low.
    assert res.volatility_history[0] < res.asset_volatility
    assert res.expected_default_frequency == pytest.approx(stats.norm.cdf(-res.distance_to_default))
    assert res.distance_to_default == pytest.approx(float(distance_to_default(V[-1], D, res.drift, res.asset_volatility)), rel=1e-6)


def test_black_cox_reflection_principle_and_limits():
    V, K, s = 100.0, 70.0, 0.2
    assert black_cox_default_probability(V, K, 1.0, 0.5 * s**2, s) == pytest.approx(2 * stats.norm.cdf(np.log(K / V) / s), rel=1e-12)
    assert black_cox_survival(V, K, 0.0, 0.05, s) == 1.0
    assert black_cox_survival(60.0, K, 1.0, 0.05, s) == 0.0
    # First passage is at least as likely as ending below the barrier.
    t = np.linspace(0.5, 10, 20)
    end_below = stats.norm.cdf((np.log(K / V) - (0.05 - 0.5 * s**2) * t) / (s * np.sqrt(t)))
    assert np.all(np.asarray(black_cox_default_probability(V, K, t, 0.05, s)) >= end_below)


@pytest.mark.parametrize("gamma", [0.0, 0.03])
def test_black_cox_matches_bridge_corrected_monte_carlo(gamma):
    V, K, T, mu, s = 100.0, 75.0, 2.0, 0.05, 0.25
    n_steps, n = 50, 20000
    t, paths = gbm_paths(V, mu, s, T=T, n_steps=n_steps, n_paths=n, rng=2)
    barrier = K * np.exp(-gamma * (T - t))
    hit = first_passage_crossing_probability(paths, barrier, s, T / n_steps)
    est, se = hit.mean(), hit.std(ddof=1) / np.sqrt(n)
    exact = black_cox_default_probability(V, K, T, mu, s, gamma=gamma)
    if gamma == 0.0:
        assert abs(est - exact) < 4 * se
    else:
        # An exponential barrier is not linear in log space on a step, so the bridge
        # correction is approximate; its error is O(gamma * dt) and below 1e-3 here.
        assert abs(est - exact) < 4 * se + 1e-3


def test_leland_smooth_pasting_and_value_identity():
    V, C, r, s, tau, alpha = 100.0, 5.0, 0.06, 0.2, 0.35, 0.5
    res = leland_model(V, C, r, s, tau, alpha)
    VB = res.default_barrier
    assert VB == pytest.approx((1 - tau) * C / (r + 0.5 * s**2))
    h = 1e-4 * VB
    # Smooth pasting: dE/dV = 0 at the endogenous barrier (one-sided difference, O(h)).
    assert (leland_model(VB + h, C, r, s, tau, alpha, VB).equity - 0.0) / h == pytest.approx(0.0, abs=1e-3)
    assert res.equity + res.debt == pytest.approx(res.firm_value)
    # Any other barrier leaves equity with a kink: lower barriers give negative slope at V_B.
    low = 0.8 * VB
    assert (leland_model(low + h, C, r, s, tau, alpha, low).equity) / h < 0


def test_leland_optimal_capital_structure():
    """Leland (1994), base case (r = 6%, sigma = 20%, tau = 35%, alpha = 50%): optimal
    leverage of about 75% and a yield spread of about 75 basis points."""
    C, res = leland_optimal_coupon(100.0, 0.06, 0.2, 0.35, 0.5)
    assert res.leverage == pytest.approx(0.75, abs=0.005)
    assert res.credit_spread == pytest.approx(0.0075, abs=0.0002)
    for dc in (-0.1, 0.1):
        assert leland_model(100.0, C + dc, 0.06, 0.2).firm_value < res.firm_value
