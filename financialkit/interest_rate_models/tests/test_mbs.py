"""Tests for Richard and Roll's prepayment model and Monte Carlo pass-through valuation."""

import numpy as np
import pytest

from financialkit.fixed_income import NelsonSiegelCurve, ShiftedCurve
from financialkit.interest_rate_models import G2PlusPlusModel, HullWhiteModel, MBSResult, VasicekModel, mbs_price, richard_roll_cpr
from financialkit.interest_rate_models.systems.mbs import SEASONALITY

CURVE = NelsonSiegelCurve(0.055, -0.015, 0.0, 2.0)


def _level_annuity(face, wac, n, cpr=0.0):
    """Deterministic pass-through flows (paid at WAC) under a constant CPR."""
    c, smm, B, flows = wac / 12, 1 - (1 - cpr) ** (1 / 12), face, []
    for k in range(1, n + 1):
        sched = B * c / (1 - (1 + c) ** -(n - k + 1)) - B * c
        prepaid = (B - sched) * smm
        flows.append(B * c + sched + prepaid)
        B -= sched + prepaid
    return np.array(flows)


def test_richard_roll_multipliers():
    full = richard_roll_cpr(0.08, 0.06, 60, 1)
    assert full == pytest.approx((0.28 + 0.14 * np.arctan(0.03)) * 0.94 * 1.0)
    assert richard_roll_cpr(0.08, 0.06, 15, 1) == pytest.approx(full / 2)  # seasoning ramps over 30 months
    assert richard_roll_cpr(0.08, 0.06, 60, 11) / full == pytest.approx(1.23 / 0.94)
    assert richard_roll_cpr(0.08, 0.06, 60, 13) == pytest.approx(full)  # months wrap
    assert richard_roll_cpr(0.08, 0.06, 60, 1, 0.0) == pytest.approx(0.3 * full)  # burnout
    cpr = richard_roll_cpr(0.08, np.linspace(0.12, 0.0, 50), 60, 8)
    assert isinstance(cpr, np.ndarray) and np.all(np.diff(cpr) > 0)
    assert 0.0 < cpr.min() and cpr.max() < 0.5 * 1.10
    assert SEASONALITY.mean() == pytest.approx(1.0, abs=0.01)
    with pytest.raises(ValueError, match="pool_factor"):
        richard_roll_cpr(0.08, 0.06, 60, 1, -0.1)


@pytest.mark.parametrize("cpr", [0.0, 0.08])
def test_constant_cpr_is_present_value_on_the_curve(cpr):
    """With a constant CPR the flows are deterministic and E[discount] = P(0, t) in a curve-fitted model."""
    model = HullWhiteModel(CURVE, 0.1, 0.01)
    res = mbs_price(model, 0.065, 0.065, n_paths=4000, prepayment=lambda *a: cpr, rng=1)
    flows = _level_annuity(100.0, 0.065, 360, cpr)
    exact = float(np.sum(flows * CURVE.discount(np.arange(1, 361) / 12)))
    assert isinstance(res, MBSResult) and res.n_paths == 4000
    assert abs(res.price - exact) < 4 * res.std_error + 1e-3  # plus the trapezoidal-rule bias of monthly discounting
    assert res.expected_cash_flows == pytest.approx(flows)
    assert res.expected_balance[0] == 100.0 and res.expected_balance[-1] == pytest.approx(0.0, abs=1e-9)
    assert res.expected_cpr == pytest.approx(cpr)


def test_servicing_spread_and_zero_wac():
    model = HullWhiteModel(CURVE, 0.1, 0.01)
    full = mbs_price(model, 0.07, 0.07, n_paths=500, rng=2)
    net = mbs_price(model, 0.07, 0.065, n_paths=500, rng=2)
    wide = mbs_price(model, 0.07, 0.07, n_paths=500, spread=0.01, rng=2)
    assert net.price < full.price and wide.price < full.price
    flat = mbs_price(model, 0.0, 0.0, term_months=12, n_paths=10, prepayment=lambda *a: 0.0, rng=0)
    assert flat.expected_cash_flows == pytest.approx(np.full(12, 100.0 / 12))


def test_negative_convexity():
    """Prepayments cap the price as rates fall: the pass-through gains less than a bond with the same schedule and no prepayment."""
    prices, bullets = [], []
    for shift in (-0.015, 0.0, 0.015):
        model = HullWhiteModel(ShiftedCurve(CURVE, shift), 0.1, 0.01)
        prices.append(mbs_price(model, 0.075, 0.07, n_paths=1000, rng=3).price)
        bullets.append(mbs_price(model, 0.075, 0.07, n_paths=1000, prepayment=lambda *a: 0.0, rng=3).price)
    assert prices[0] > prices[1] > prices[2]
    assert prices[0] - prices[1] < bullets[0] - bullets[1]
    assert prices[0] - 2 * prices[1] + prices[2] < 0 < bullets[0] - 2 * bullets[1] + bullets[2]


def test_burnout_and_refinancing_waves():
    """Rates that start low trigger a refinancing wave early, then burnout lowers the CPR."""
    low = mbs_price(VasicekModel(0.03, 0.2, 0.06, 0.005), 0.08, 0.075, n_paths=300, age=30, rng=4)
    high = mbs_price(VasicekModel(0.08, 0.2, 0.06, 0.005), 0.08, 0.075, n_paths=300, age=30, rng=4)
    assert low.expected_cpr[:12].mean() > high.expected_cpr[:12].mean()
    assert low.expected_balance[60] < high.expected_balance[60]


def test_mbs_validation():
    with pytest.raises(ValueError, match="one-factor"):
        mbs_price(G2PlusPlusModel(CURVE, 0.1, 0.01, 0.3, 0.005, -0.5), 0.07, 0.07)
    model = HullWhiteModel(CURVE, 0.1, 0.01)
    with pytest.raises(ValueError, match="term_months"):
        mbs_price(model, 0.07, 0.07, term_months=0)
    with pytest.raises(ValueError, match="start_month"):
        mbs_price(model, 0.07, 0.07, start_month=13)
