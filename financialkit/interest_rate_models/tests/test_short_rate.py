"""Tests for Merton, Vasicek and CIR, the Feller condition and Feynman-Kac Monte Carlo."""

import numpy as np
import pytest
from scipy.integrate import quad, trapezoid
from scipy.stats import gamma

from financialkit.constants import DEFAULT_RTOL
from financialkit.interest_rate_models import CIRModel, MertonShortRateModel, VasicekModel, feller_condition, monte_carlo_bond_prices

MATURITIES = [0.5, 1.0, 3.0, 7.0]


@pytest.mark.parametrize(
    "model",
    [
        MertonShortRateModel(0.03, 0.002, 0.01),
        VasicekModel(0.03, 0.5, 0.05, 0.015),
        CIRModel(0.03, 0.5, 0.05, 0.1),
    ],
    ids=["merton", "vasicek", "cir"],
)
def test_closed_form_bonds_match_feynman_kac_monte_carlo(model):
    res = monte_carlo_bond_prices(model, MATURITIES, n_steps=140, n_paths=20_000, rng=7)
    assert np.all(np.abs(res.z_scores) < 4)


def test_cir_euler_full_truncation_matches_closed_form():
    model = CIRModel(0.03, 0.5, 0.05, 0.1)
    times, r = model.simulate(5.0, 250, 20_000, rng=3, method="euler")
    assert np.all(r >= 0)
    disc = np.exp(-trapezoid(r, times, axis=1))
    se = disc.std(ddof=1) / np.sqrt(disc.size)
    assert abs(disc.mean() - model.zero_coupon_bond(5.0)) < 4 * se


def test_vasicek_solves_the_bond_pricing_pde():
    """P(t, T; r) satisfies P_t + kappa (theta - r) P_r + sigma^2 / 2 P_rr - r P = 0."""
    m = VasicekModel(0.04, 0.3, 0.06, 0.02)
    T, t, r, h = 8.0, 1.0, 0.045, 1e-4
    P = lambda t, r: m.zero_coupon_bond(T, t=t, state=r)
    Pt = (P(t + h, r) - P(t - h, r)) / (2 * h)
    Pr = (P(t, r + h) - P(t, r - h)) / (2 * h)
    Prr = (P(t, r + h) - 2 * P(t, r) + P(t, r - h)) / h**2
    assert Pt + m.kappa * (m.theta - r) * Pr + 0.5 * m.sigma**2 * Prr - r * P(t, r) == pytest.approx(0.0, abs=1e-6)


def test_cir_solves_the_bond_pricing_pde():
    m = CIRModel(0.04, 0.3, 0.06, 0.1)
    T, t, r, h = 8.0, 1.0, 0.045, 1e-4
    P = lambda t, r: m.zero_coupon_bond(T, t=t, state=r)
    Pt = (P(t + h, r) - P(t - h, r)) / (2 * h)
    Pr = (P(t, r + h) - P(t, r - h)) / (2 * h)
    Prr = (P(t, r + h) - 2 * P(t, r) + P(t, r - h)) / h**2
    assert Pt + m.kappa * (m.theta - r) * Pr + 0.5 * m.sigma**2 * r * Prr - r * P(t, r) == pytest.approx(0.0, abs=1e-6)


def test_vasicek_long_yield_and_zero_volatility_limit():
    m = VasicekModel(0.02, 0.4, 0.06, 0.02)
    assert m.yields(1e4) == pytest.approx(m.long_yield, abs=2e-5)  # converges like 1/T
    det = VasicekModel(0.02, 0.4, 0.06, 0.0)
    integral = quad(lambda s: 0.06 + (0.02 - 0.06) * np.exp(-0.4 * s), 0, 5)[0]
    assert det.zero_coupon_bond(5.0) == pytest.approx(np.exp(-integral), rel=DEFAULT_RTOL)


def test_merton_yields_turn_negative_at_long_maturities():
    """The sigma^2 tau^3 / 6 convexity term eventually dominates: R(T) = r - mu T/2 - sigma^2 T^2 / 6."""
    m = MertonShortRateModel(0.05, 0.0, 0.02)
    T = np.array([1.0, 10.0, 60.0])
    assert np.allclose(m.yields(T), 0.05 - 0.02**2 * T**2 / 6)
    assert m.yields(60.0) < 0


def test_vasicek_simulation_has_exact_ou_moments():
    m = VasicekModel(0.02, 0.8, 0.05, 0.02)
    _, r = m.simulate(2.0, 4, 50_000, rng=1)
    mean = 0.05 + (0.02 - 0.05) * np.exp(-1.6)
    var = 0.02**2 * (1 - np.exp(-3.2)) / 1.6
    assert abs(r[:, -1].mean() - mean) < 4 * np.sqrt(var / r.shape[0])
    assert r[:, -1].var() == pytest.approx(var, rel=0.03)


def test_feller_condition_and_cir_stationary_law():
    assert feller_condition(0.5, 0.05, 0.2) == pytest.approx(1.25)
    assert np.allclose(feller_condition(np.array([0.5, 1.0]), 0.04, 0.2), [1.0, 2.0])
    m = CIRModel(0.05, 2.0, 0.05, 0.15)
    shape, scale = m.stationary_shape_scale()
    _, r = m.simulate(10.0, 20, 40_000, rng=2)
    x = r[:, -1]
    assert abs(x.mean() - gamma.mean(shape, scale=scale)) < 4 * gamma.std(shape, scale=scale) / np.sqrt(x.size)


def test_cir_exact_paths_touch_zero_only_when_feller_fails():
    kept = CIRModel(0.04, 2.0, 0.04, 0.2)  # ratio 4
    broken = CIRModel(0.04, 1.0, 0.04, 0.5)  # ratio 0.32
    assert kept.feller_ratio >= 1 > broken.feller_ratio
    _, a = kept.simulate(5.0, 500, 2000, rng=4)
    _, b = broken.simulate(5.0, 500, 2000, rng=4)
    assert a.min() > 1e-4
    assert np.mean(b.min(axis=1) < 1e-6) > 0.5


def test_validation():
    with pytest.raises(ValueError):
        VasicekModel(0.03, 0.0, 0.05, 0.01)
    with pytest.raises(ValueError):
        CIRModel(-0.01, 0.5, 0.05, 0.1)
    with pytest.raises(ValueError):
        CIRModel(0.03, 0.5, 0.05, 0.1).simulate(1.0, 10, 10, method="milstein")
    with pytest.raises(ValueError):
        monte_carlo_bond_prices(VasicekModel(0.03, 0.5, 0.05, 0.01), [0.0, 1.0])


def test_short_rate_model_reprs_round_trip():
    namespace = {"MertonShortRateModel": MertonShortRateModel, "VasicekModel": VasicekModel, "CIRModel": CIRModel}
    for model in (MertonShortRateModel(0.03, 0.001, 0.01), VasicekModel(0.03, 0.4, 0.05, 0.015), CIRModel(0.03, 0.4, 0.05, 0.12)):
        rebuilt = eval(repr(model), namespace)
        assert rebuilt.zero_coupon_bond(5.0) == model.zero_coupon_bond(5.0)
    curve = VasicekModel(0.03, 0.4, 0.05, 0.015).curve()
    assert repr(curve).startswith("ModelCurve(VasicekModel(")


def test_merton_bond_option_volatility_is_ho_lee():
    """Merton's arithmetic Brownian rate is Ho-Lee on a curve it fits exactly: both give sigma (S - T) sqrt(T)."""
    T, S = np.array([0.5, 1.0, 2.0]), np.array([1.0, 5.0, 10.0])
    np.testing.assert_allclose(MertonShortRateModel(0.03, 0.0, 0.01).bond_option_volatility(T, S), 0.01 * (S - T) * np.sqrt(T), rtol=1e-15)


def test_short_rate_models_reject_invalid_parameters():
    with pytest.raises(ValueError, match="sigma must be non-negative"):
        MertonShortRateModel(0.03, 0.0, -0.01)
    with pytest.raises(ValueError, match="sigma must be non-negative"):
        VasicekModel(0.03, 0.4, 0.05, -0.01)
    with pytest.raises(ValueError, match="theta must be positive, got 0"):
        CIRModel(0.03, 0.4, 0.0, 0.1)
