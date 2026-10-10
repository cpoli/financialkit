"""Tests for zero-coupon and coupon bond options, Jamshidian swaptions, caps, floors and Black swaptions."""

import numpy as np
import pytest
from scipy.stats import norm

from financialkit.constants import DEFAULT_RTOL
from financialkit.fixed_income import NelsonSiegelCurve, swap_rate
from financialkit.interest_rate_models import (
    G2PlusPlusModel,
    HoLeeModel,
    HullWhiteModel,
    VasicekModel,
    black_swaption_price,
    cap_floor_price,
    coupon_bond_option,
    forward_swap_annuity,
    forward_swap_rate,
    jamshidian_swaption,
    model_cap_floor_price,
    path_discount_factors,
    swaption_schedule,
    zero_coupon_bond_option,
)
from financialkit.options import implied_volatility

CURVE = NelsonSiegelCurve(0.05, -0.02, 0.015, 2.0)
HW = HullWhiteModel(CURVE, 0.1, 0.01)


def _simulate_to(model, T, n_paths=40_000, rng=0):
    times, r = model.simulate(T, 200, n_paths, rng=rng)
    return path_discount_factors(times, r)[:, -1], r[:, -1]


@pytest.mark.parametrize(
    "model",
    [VasicekModel(0.03, 0.4, 0.05, 0.015), HoLeeModel(CURVE, 0.01), HW, G2PlusPlusModel(CURVE, 0.6, 0.012, 0.06, 0.008, -0.7)],
    ids=lambda m: type(m).__name__,
)
def test_zero_coupon_bond_option_parity(model):
    K = np.array([0.7, 0.8, 0.9])
    c = zero_coupon_bond_option(model, 2.0, 6.0, K, "call")
    p = zero_coupon_bond_option(model, 2.0, 6.0, K, "put")
    assert np.allclose(c - p, model.zero_coupon_bond(6.0) - K * model.zero_coupon_bond(2.0), rtol=DEFAULT_RTOL)


def test_zero_coupon_bond_option_matches_monte_carlo():
    model = VasicekModel(0.03, 0.4, 0.05, 0.015)
    D, rT = _simulate_to(model, 2.0, rng=1)
    K = float(model.zero_coupon_bond(6.0) / model.zero_coupon_bond(2.0))
    pay = D * np.maximum(model.zero_coupon_bond(6.0, t=2.0, state=rT) - K, 0.0)
    assert abs(pay.mean() - zero_coupon_bond_option(model, 2.0, 6.0, K)) < 4 * pay.std(ddof=1) / np.sqrt(pay.size)


def test_coupon_bond_option_decomposition_matches_monte_carlo():
    times = np.array([3.0, 4.0, 5.0, 6.0])
    flows = np.array([5.0, 5.0, 5.0, 105.0])
    res = coupon_bond_option(HW, 2.0, times, flows, 100.0, "put")
    assert res.price == pytest.approx(res.components.sum())
    assert float(np.sum(flows * HW.zero_coupon_bond(times, t=2.0, state=res.critical_rate))) == pytest.approx(100.0)
    assert np.allclose(res.strikes, HW.zero_coupon_bond(times, t=2.0, state=res.critical_rate))
    D, rT = _simulate_to(HW, 2.0, rng=2)
    bond = (flows * HW.zero_coupon_bond(times[None, :], t=2.0, state=rT[:, None])).sum(axis=1)
    pay = D * np.maximum(100.0 - bond, 0.0)
    assert abs(pay.mean() - res.price) < 4 * pay.std(ddof=1) / np.sqrt(pay.size)


def test_jamshidian_swaption_matches_monte_carlo_and_parity():
    expiry, tenor, K = 2.0, 5.0, 0.05
    payer = jamshidian_swaption(HW, expiry, tenor, K, freq=2, payer=True)
    receiver = jamshidian_swaption(HW, expiry, tenor, K, freq=2, payer=False)
    A = forward_swap_annuity(CURVE, expiry, tenor, 2)
    S = forward_swap_rate(CURVE, expiry, tenor, 2)
    assert payer.price - receiver.price == pytest.approx(A * (S - K), rel=1e-8)
    D, rT = _simulate_to(HW, expiry, rng=3)
    sched = swaption_schedule(expiry, tenor, 2)
    P = HW.zero_coupon_bond(sched[None, :], t=expiry, state=rT[:, None])
    pay = D * np.maximum(1.0 - (K * 0.5 * P.sum(axis=1) + P[:, -1]), 0.0)
    assert abs(pay.mean() - payer.price) < 4 * pay.std(ddof=1) / np.sqrt(pay.size)
    assert jamshidian_swaption(HW, expiry, tenor, K, notional=1e6).price == pytest.approx(1e6 * payer.price)


def test_jamshidian_needs_a_one_factor_model():
    with pytest.raises(ValueError):
        jamshidian_swaption(G2PlusPlusModel(CURVE, 0.6, 0.012, 0.06, 0.008, -0.7), 1.0, 2.0, 0.05)


def test_black_swaption_hull_example_29_5():
    """Hull, 10th ed., Example 29.5: 5 x 3 payer at 6.2%, flat 6% curve, 20% vol, principal 100: 2.07."""
    assert black_swaption_price(0.06, 5.0, 3.0, 0.062, 0.2, 2, notional=100.0) == pytest.approx(2.07, abs=5e-3)


def test_black_swaption_parity_and_spot_start_swap_rate():
    K = np.array([0.03, 0.05, 0.07])
    pay = black_swaption_price(CURVE, 3.0, 5.0, K, 0.2)
    rec = black_swaption_price(CURVE, 3.0, 5.0, K, 0.2, payer=False)
    A, S = forward_swap_annuity(CURVE, 3.0, 5.0), forward_swap_rate(CURVE, 3.0, 5.0)
    assert np.allclose(pay - rec, A * (S - K), rtol=1e-10)
    # A swap starting now (expiry -> 0) has the fixed_income par swap rate.
    assert forward_swap_rate(CURVE, 1e-12, 5.0) == pytest.approx(float(swap_rate(CURVE, 5.0)), rel=1e-8)


def test_cap_floor_parity_and_intrinsic_limit():
    K = 0.045
    cap = cap_floor_price(CURVE, K, 5.0, 0.2, freq=4)
    floor = cap_floor_price(CURVE, K, 5.0, 0.2, freq=4, kind="floor")
    tau = cap.payment_times - cap.reset_times
    swap = np.sum(tau * CURVE.discount(cap.payment_times) * (cap.forwards - K))
    assert cap.price - floor.price == pytest.approx(swap, rel=1e-10)
    assert cap.reset_times[0] == pytest.approx(0.25) and cap.payment_times[-1] == pytest.approx(5.0)
    tiny = cap_floor_price(CURVE, K, 5.0, 1e-9, freq=4)
    assert np.allclose(tiny.caplets, tau * CURVE.discount(cap.payment_times) * np.maximum(cap.forwards - K, 0.0))


def test_caplet_is_black76_on_the_forward():
    """Hull, 10th ed., Example 29.3 setup: caplet = L tau P(0, t2) Black(F, K, sigma, t1)."""
    res = cap_floor_price(CURVE, 0.05, 1.5, np.array([0.2, 0.25, 0.3, 0.35, 0.4]), freq=4, notional=10_000)
    assert res.caplets.size == 5
    k = 2
    F, t1, t2 = res.forwards[k], res.reset_times[k], res.payment_times[k]
    vol = 0.3
    d1 = (np.log(F / 0.05) + 0.5 * vol**2 * t1) / (vol * np.sqrt(t1))
    expected = 10_000 * 0.25 * CURVE.discount(t2) * (F * norm.cdf(d1) - 0.05 * norm.cdf(d1 - vol * np.sqrt(t1)))
    assert res.caplets[k] == pytest.approx(float(expected), rel=1e-10)


def test_model_caps_parity_and_monte_carlo():
    K = 0.045
    cap = model_cap_floor_price(HW, K, 3.0, freq=2)
    floor = model_cap_floor_price(HW, K, 3.0, freq=2, kind="floor")
    tau = cap.payment_times - cap.reset_times
    assert cap.price - floor.price == pytest.approx(np.sum(tau * CURVE.discount(cap.payment_times) * (cap.forwards - K)), rel=1e-8)
    # Last caplet by simulation: pays tau max(L - K, 0) at t2, worth tau P(t1, t2) max(L - K, 0) at t1.
    t1, t2 = cap.reset_times[-1], cap.payment_times[-1]
    D, rT = _simulate_to(HW, t1, rng=5)
    P = HW.zero_coupon_bond(t2, t=t1, state=rT)
    L = (1 / P - 1) / (t2 - t1)
    pay = D * (t2 - t1) * P * np.maximum(L - K, 0.0)
    assert abs(pay.mean() - cap.caplets[-1]) < 4 * pay.std(ddof=1) / np.sqrt(pay.size)


def test_hull_white_caplets_have_a_black_implied_vol():
    cap = model_cap_floor_price(HW, 0.045, 5.0, freq=4)
    k = 8
    t1, t2 = cap.reset_times[k], cap.payment_times[k]
    undiscounted = cap.caplets[k] / ((t2 - t1) * CURVE.discount(t2))
    vol = implied_volatility(undiscounted, cap.forwards[k], 0.045, t1, 0.0, 0.0)
    assert 0.05 < vol < 1.0
    black = cap_floor_price(CURVE, 0.045, 5.0, vol, freq=4).caplets[k]
    assert black == pytest.approx(cap.caplets[k], rel=1e-7)


def test_validation():
    with pytest.raises(ValueError):
        zero_coupon_bond_option(HW, 3.0, 2.0, 0.9)
    with pytest.raises(ValueError):
        cap_floor_price(CURVE, 0.05, 2.0, 0.2, kind="collar")
    with pytest.raises(ValueError):
        cap_floor_price(CURVE, 0.05, 0.25, 0.2, freq=4)
    with pytest.raises(ValueError):
        coupon_bond_option(HW, 2.0, [1.0, 3.0], [5.0, 105.0], 100.0)
