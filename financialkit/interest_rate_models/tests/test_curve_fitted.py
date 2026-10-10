"""Tests for Ho-Lee, Hull-White, Black-Derman-Toy and the futures convexity adjustment."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.fixed_income import FlatCurve, NelsonSiegelCurve, ZeroCurve
from financialkit.interest_rate_models import (
    HoLeeModel,
    HullWhiteModel,
    MertonShortRateModel,
    VasicekModel,
    black_derman_toy_tree,
    futures_convexity_adjustment,
    monte_carlo_bond_prices,
    path_discount_factors,
)

CURVE = NelsonSiegelCurve(0.05, -0.02, 0.015, 2.0)
T = np.array([0.5, 2.0, 5.0, 10.0])


@pytest.mark.parametrize("model", [HoLeeModel(CURVE, 0.01), HullWhiteModel(CURVE, 0.15, 0.012)], ids=["ho_lee", "hull_white"])
def test_models_reprice_the_initial_curve_exactly_and_by_simulation(model):
    assert np.allclose(model.zero_coupon_bond(T), CURVE.discount(T), rtol=DEFAULT_RTOL)
    res = monte_carlo_bond_prices(model, T, n_steps=200, n_paths=20_000, rng=11)
    assert np.all(np.abs(res.z_scores) < 4)


def test_hull_white_on_a_vasicek_curve_is_vasicek():
    v = VasicekModel(0.03, 0.5, 0.05, 0.01)
    hw = HullWhiteModel(v.curve(), 0.5, 0.01)
    assert hw.zero_coupon_bond(9.0, t=2.0, state=0.045) == pytest.approx(float(v.zero_coupon_bond(9.0, t=2.0, state=0.045)), rel=1e-7)
    assert hw.bond_option_volatility(2.0, 9.0) == pytest.approx(float(v.bond_option_volatility(2.0, 9.0)))
    assert np.allclose(hw.theta(np.array([0.5, 3.0])), 0.5 * 0.05, atol=1e-6)


def test_ho_lee_on_a_merton_curve_is_merton():
    m = MertonShortRateModel(0.03, 0.002, 0.01)
    hl = HoLeeModel(m.curve(), 0.01)
    assert hl.zero_coupon_bond(6.0, t=1.5, state=0.04) == pytest.approx(float(m.zero_coupon_bond(6.0, t=1.5, state=0.04)), rel=1e-7)
    assert np.allclose(hl.theta(np.array([1.0, 4.0])), 0.002, atol=1e-6)


def test_hull_white_tends_to_ho_lee_as_mean_reversion_vanishes():
    hl = HoLeeModel(CURVE, 0.01)
    hw = HullWhiteModel(CURVE, 1e-7, 0.01)
    assert hw.zero_coupon_bond(8.0, t=3.0, state=0.05) == pytest.approx(float(hl.zero_coupon_bond(8.0, t=3.0, state=0.05)), rel=1e-6)


def test_ho_lee_theta_on_a_flat_curve():
    """theta(t) = f_T(0, t) + sigma^2 t, which is sigma^2 t on a flat curve."""
    t = np.array([0.5, 2.0, 7.0])
    assert np.allclose(HoLeeModel(0.04, 0.01).theta(t), 0.01**2 * t, atol=1e-8)


def test_future_bond_prices_are_martingales_after_discounting():
    """E[D(0, t) P(t, T)] = P(0, T) under Hull-White."""
    hw = HullWhiteModel(CURVE, 0.15, 0.012)
    times, r = hw.simulate(3.0, 150, 20_000, rng=2)
    disc = path_discount_factors(times, r)[:, -1] * hw.zero_coupon_bond(10.0, t=3.0, state=r[:, -1])
    assert abs(disc.mean() - CURVE.discount(10.0)) < 4 * disc.std(ddof=1) / np.sqrt(disc.size)


def test_bdt_reprices_the_curve_and_has_lognormal_spacing():
    tree = black_derman_toy_tree(CURVE, 0.18, 6.0, 60)
    for Tk in (0.1, 1.0, 3.5, 6.0):
        assert tree.zero_coupon_bond(Tk) == pytest.approx(float(CURVE.discount(Tk)), rel=1e-9)
    ratio = tree.rates[10, 1:11] / tree.rates[10, :10]
    assert np.allclose(ratio, np.exp(2 * 0.18 * np.sqrt(tree.dt)))
    assert np.all(tree.rates[np.isfinite(tree.rates)] > 0)
    assert tree.arrow_debreu[1, :2].sum() == pytest.approx(float(CURVE.discount(0.1)))


def test_bdt_on_a_zero_curve_with_term_structure_of_vols():
    curve = ZeroCurve([1.0, 2.0, 3.0, 4.0], [0.03, 0.035, 0.04, 0.042])
    tree = black_derman_toy_tree(curve, np.linspace(0.25, 0.15, 8), 4.0, 8)
    assert tree.zero_coupon_bond(4.0) == pytest.approx(float(curve.discount(4.0)), rel=1e-9)


def test_bdt_bond_options():
    tree = black_derman_toy_tree(FlatCurve(0.05), 1e-8, 5.0, 50)
    fwd = np.exp(-0.05 * 3)
    # With no volatility the European option is the discounted intrinsic value of the forward.
    assert tree.bond_option(2.0, 5.0, 0.8, "call") == pytest.approx(np.exp(-0.1) * (fwd - 0.8), rel=1e-6)
    tree = black_derman_toy_tree(CURVE, 0.2, 5.0, 50)
    euro = tree.bond_option(2.0, 5.0, 0.85, "put")
    amer = tree.bond_option(2.0, 5.0, 0.85, "put", american=True)
    call = tree.bond_option(2.0, 5.0, 0.85, "call")
    assert amer >= euro > 0
    assert call - euro == pytest.approx(tree.zero_coupon_bond(5.0) - 0.85 * tree.zero_coupon_bond(2.0), rel=1e-9)
    with pytest.raises(ValueError):
        tree.bond_option(5.0, 2.0, 0.9)
    with pytest.raises(ValueError):
        tree.zero_coupon_bond(1.234)


def test_futures_convexity_hull_example_6_4():
    """Hull, 10th ed., Example 6.4: sigma = 0.012, t1 = 8, t2 = 8.25 gives 0.475%."""
    assert futures_convexity_adjustment(8.0, 8.25, 0.012) == pytest.approx(0.004752)
    assert futures_convexity_adjustment(8.0, 8.25, 0.012, a=1e-8) == pytest.approx(0.004752, rel=1e-6)
    assert futures_convexity_adjustment(8.0, 8.25, 0.012, a=0.05) < 0.004752


@pytest.mark.parametrize("a", [0.0, 0.1])
def test_futures_rate_is_the_risk_neutral_expected_rate(a):
    """E^Q[R(t1, t2)] - F(0; t1, t2) equals the convexity adjustment (continuous compounding)."""
    t1, t2, sigma = 4.0, 4.25, 0.015
    model = HullWhiteModel(CURVE, a, sigma) if a else HoLeeModel(CURVE, sigma)
    _, r = model.simulate(t1, 8, 40_000, rng=9)
    R = -np.log(model.zero_coupon_bond(t2, t=t1, state=r[:, -1])) / (t2 - t1)
    forward = float(CURVE.forward_rate(t1, t2))
    expected = futures_convexity_adjustment(t1, t2, sigma, a)
    assert abs(R.mean() - forward - expected) < 4 * R.std(ddof=1) / np.sqrt(R.size)


def test_validation():
    with pytest.raises(ValueError):
        HullWhiteModel(CURVE, -0.1, 0.01)
    with pytest.raises(ValueError):
        HoLeeModel(CURVE, -0.01)
    with pytest.raises(ValueError):
        black_derman_toy_tree(CURVE, 0.2, 5.0, 0)
    with pytest.raises(ValueError):
        futures_convexity_adjustment(2.0, 1.0, 0.01)
