"""Tests for hazard curves, Jarrow-Turnbull bonds and Duffie-Singleton recovery."""

import numpy as np
import pytest
from scipy import stats

from financialkit.constants import DEFAULT_RTOL
from financialkit.credit import (
    HazardCurve,
    duffie_singleton_cir_zero,
    hazard_from_spread,
    jarrow_turnbull_zero,
    risky_zero_coupon,
    spread_from_hazard,
)
from financialkit.interest_rate_models import CIRModel


def test_hazard_curve_piecewise_integral_and_inverse():
    c = HazardCurve([1.0, 3.0, 5.0], [0.01, 0.02, 0.04])
    t = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 7.0])
    expected = np.array([0.0, 0.005, 0.01, 0.03, 0.05, 0.09, 0.05 + 0.04 * 4])
    np.testing.assert_allclose(c.cumulative_hazard(t), expected, rtol=DEFAULT_RTOL, atol=1e-15)
    np.testing.assert_allclose(c.inverse_cumulative_hazard(expected[1:]), t[1:], rtol=1e-12)
    np.testing.assert_allclose(c.hazard([0.5, 1.0, 1.5, 9.0]), [0.01, 0.01, 0.02, 0.04])
    assert c.default_probability(1.0, 3.0) == pytest.approx(np.exp(-0.01) - np.exp(-0.05))
    with pytest.raises(ValueError):
        HazardCurve([1.0, 0.5], [0.01, 0.02])
    with pytest.raises(ValueError):
        HazardCurve([1.0], [-0.01])


def test_zero_hazard_never_defaults():
    c = HazardCurve([1.0, 2.0], [0.05, 0.0])
    assert np.isinf(c.inverse_cumulative_hazard(0.2))
    assert c.inverse_cumulative_hazard(0.05) == pytest.approx(1.0)


def test_sampled_default_times_are_exponential():
    c = HazardCurve.flat(0.1)
    tau = c.sample_default_times(5000, rng=0)
    assert stats.kstest(tau, "expon", args=(0, 10.0)).pvalue > 1e-3
    pc = HazardCurve([2.0, 4.0], [0.05, 0.2])
    tau = pc.sample_default_times(20000, rng=1)
    p = pc.default_probability(4.0)
    est = np.mean(tau <= 4.0)
    assert abs(est - p) < 4 * np.sqrt(p * (1 - p) / tau.size)


def test_jarrow_turnbull_and_credit_triangle():
    c = HazardCurve.flat(0.02)
    assert jarrow_turnbull_zero(5.0, 0.05, c, 0.0) == pytest.approx(np.exp(-0.25 - 0.1))
    assert jarrow_turnbull_zero(5.0, 0.05, c, 1.0) == pytest.approx(np.exp(-0.25))
    assert hazard_from_spread(spread_from_hazard(0.03, 0.35), 0.35) == pytest.approx(0.03)


def test_recovery_conventions_order():
    # Without recovery or without default the three conventions coincide.
    T, r, h = 5.0, 0.05, 0.03
    for conv in ("rmv", "rt", "rfv"):
        assert risky_zero_coupon(T, r, h, 0.0, conv) == pytest.approx(np.exp(-(r + h) * T))
        assert risky_zero_coupon(T, r, 0.0, 0.4, conv) == pytest.approx(np.exp(-r * T))
    # Recovery of face value with R = 1 is a riskless claim paid early.
    assert risky_zero_coupon(T, r, h, 1.0, "rfv") > np.exp(-r * T)
    with pytest.raises(ValueError):
        risky_zero_coupon(T, r, h, 0.4, "bad")


def test_duffie_singleton_cir_against_monte_carlo():
    """RMV price E[exp(-int (r + L lambda))] by simulating the CIR intensity."""
    r, lam0, k, th, s, R, T = 0.03, 0.02, 0.5, 0.04, 0.1, 0.4, 5.0
    exact = duffie_singleton_cir_zero(T, r, lam0, k, th, s, R)
    n_steps, n = 200, 20000
    _, lam = CIRModel(lam0, k, th, s).simulate(T, n_steps, n, rng=3)
    integral = np.trapezoid(lam, dx=T / n_steps, axis=1) if hasattr(np, "trapezoid") else np.trapz(lam, dx=T / n_steps, axis=1)
    x = np.exp(-r * T - (1 - R) * integral)
    # Trapezoid bias is O(dt^2) on a smooth integrand, negligible next to the standard error.
    assert abs(x.mean() - exact) < 4 * x.std(ddof=1) / np.sqrt(n) + 1e-5
