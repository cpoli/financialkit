"""Tests for Hull and White's trinomial tree, callable bonds and the option-adjusted spread."""

import numpy as np
import pytest

from financialkit.fixed_income import NelsonSiegelCurve, ShiftedCurve, ZeroCurve
from financialkit.fixed_income.core.base import coupon_times
from financialkit.interest_rate_models import (
    HullWhiteModel,
    callable_bond_price,
    coupon_bond_option,
    hull_white_tree,
    option_adjusted_spread,
)

CURVE = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)
HULL_CURVE = ZeroCurve([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [0.0343, 0.03824, 0.04183, 0.04512, 0.04812, 0.05086])


def test_hull_three_step_tree():
    """Hull, *Options, Futures, and Other Derivatives*, 10th ed., Sec. 32.7 (Figures 32.8-32.9).

    a = 0.1, sigma = 0.01, yearly steps: dR = 0.01732, j_max = 2, node A
    3.824%, nodes B-D 6.937/5.205/3.473%, E and I 9.716/2.788%, the
    branch probabilities at A and E, and Q at the second level.
    """
    tree = hull_white_tree(HULL_CURVE, 0.1, 0.01, 3.0, 3)
    assert tree.j_max == 2 and tree.dx == pytest.approx(0.01732, abs=5e-6)
    rates = tree.rates
    assert rates[0, 2] == pytest.approx(0.03824, abs=5e-6)
    assert rates[1, 1:4] == pytest.approx([0.03473, 0.05205, 0.06937], abs=5e-6)
    assert rates[2, [0, 4]] == pytest.approx([0.02788, 0.09716], abs=5e-6)
    assert np.isnan(rates[0, 0]) and np.isnan(rates[1, 4])
    assert tree.probabilities[2] == pytest.approx([1 / 6, 2 / 3, 1 / 6])
    assert tree.probabilities[4] == pytest.approx([0.0867, 0.0267, 0.8867], abs=5e-5)  # E branches to j - 2, j - 1, j
    assert tree.arrow_debreu[1, 1:4] == pytest.approx([0.1604, 0.6417, 0.1604], abs=5e-5)


@pytest.mark.parametrize("a", [0.0, 0.03, 0.2])
def test_tree_reprices_the_curve(a):
    tree = hull_white_tree(CURVE, a, 0.012, 10.0, 120)
    assert tree.probabilities.sum(axis=1) == pytest.approx(1.0)
    assert np.all(tree.probabilities >= 0)
    for T in (0.25, 3.0, 10.0):
        assert tree.zero_coupon_bond(T) == pytest.approx(float(CURVE.discount(T)), rel=1e-12)
    assert tree.arrow_debreu[-1].sum() == pytest.approx(float(CURVE.discount(10.0)), rel=1e-12)
    assert tree.width == (120 if a == 0 else tree.j_max)


def test_spread_shifts_every_discount_factor():
    """Adding s to every node rate discounts a bond at t by an extra e^{-s t}, so the straight bond is priced at its Z-spread."""
    tree = hull_white_tree(CURVE, 0.1, 0.01, 10.0, 80)
    res = callable_bond_price(tree, 10.0, 0.05, first_call=2.0, spread=0.013)
    times = coupon_times(10.0, 2)
    flows = np.full(times.size, 0.025)
    flows[-1] += 1.0
    assert res.straight_price == pytest.approx(float(np.sum(flows * ShiftedCurve(CURVE, 0.013).discount(times))), rel=1e-12)
    assert tree.zero_coupon_bond(5.0, spread=0.013) == pytest.approx(float(CURVE.discount(5.0)) * np.exp(-0.013 * 5.0), rel=1e-12)
    assert res.spread == 0.013


def test_single_call_date_is_jamshidian():
    """Callable once, at 5 years, the bond is the straight bond minus Jamshidian's European call on its remaining flows."""
    times = coupon_times(10.0, 2)
    flows = np.full(times.size, 0.025)
    flows[-1] += 1.0
    after = times > 5.0
    european = coupon_bond_option(HullWhiteModel(CURVE, 0.1, 0.01), 5.0, times[after], flows[after], 1.0, "call").price
    exact = float(np.sum(flows * CURVE.discount(times))) - european
    tree = hull_white_tree(CURVE, 0.1, 0.01, 10.0, 400)
    res = callable_bond_price(tree, 10.0, 0.05, first_call=5.0, last_call=5.0)
    assert res.call_times.tolist() == [5.0]
    # The tree is first order in dt, and the strike's position between nodes adds an oscillation of about 1e-5.
    assert res.price == pytest.approx(exact, abs=5e-5)
    assert res.option_value == pytest.approx(european, abs=5e-5)


def test_call_lowers_value_and_caps_it():
    tree = hull_white_tree(ShiftedCurve(CURVE, -0.03), 0.1, 0.01, 10.0, 80)
    curve = ShiftedCurve(CURVE, -0.03)
    bermudan = callable_bond_price(tree, 10.0, 0.06, first_call=1.0)
    european = callable_bond_price(tree, 10.0, 0.06, first_call=1.0, last_call=1.0)
    assert bermudan.price < european.price < european.straight_price
    # Worth at most the coupons up to the first call plus the call price then.
    assert bermudan.price < 0.03 * curve.discount(0.5) + 1.03 * curve.discount(1.0)
    # The issuer calls when rates are low: there is a highest calling rate on every date.
    assert np.all(np.isfinite(bermudan.call_boundary))
    never = callable_bond_price(tree, 10.0, 0.06, call_price=5.0)
    assert np.all(np.isnan(never.call_boundary)) and never.price == pytest.approx(never.straight_price)


def test_option_adjusted_spread_round_trip_and_z_spread():
    tree = hull_white_tree(CURVE, 0.1, 0.01, 10.0, 80)
    for s in (-0.12, 0.004, 0.15):  # outside the first bracket on both sides
        p = callable_bond_price(tree, 10.0, 0.05, first_call=2.0, spread=s).price
        assert option_adjusted_spread(tree, p, 10.0, 0.05, first_call=2.0) == pytest.approx(s, abs=1e-9)
    # The OAS is below the Z-spread of the same price by the value of the call.
    p = callable_bond_price(tree, 10.0, 0.05, first_call=2.0, spread=0.01).price
    z = option_adjusted_spread(tree, p, 10.0, 0.05, call_price=10.0)
    assert z > 0.01


def test_tree_validation():
    with pytest.raises(ValueError, match="n_steps"):
        hull_white_tree(CURVE, 0.1, 0.01, 10.0, 0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        hull_white_tree(CURVE, 0.1, 0.0, 10.0, 10)
    with pytest.raises(ValueError, match="negative branch probabilities"):
        hull_white_tree(CURVE, 2.0, 0.01, 10.0, 10)  # a dt = 2 > 1.816
    tree = hull_white_tree(CURVE, 0.1, 0.01, 10.0, 80)
    with pytest.raises(ValueError, match="node date"):
        callable_bond_price(tree, 10.0, 0.05, freq=3)
    with pytest.raises(ValueError, match="price must be positive"):
        option_adjusted_spread(tree, 0.0, 10.0, 0.05)
