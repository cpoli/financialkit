"""Tests for Lyashenko and Mercurio's forward market model of backward-looking rates."""

import numpy as np
import pytest

from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import (
    HullWhiteModel,
    cap_floor_price,
    fmm_caplet_price,
    fmm_volatility_decay,
    path_discount_factors,
    simulate_fmm_rate,
)

CURVE = NelsonSiegelCurve(0.04, -0.01, 0.0, 2.0)


def test_volatility_decay():
    g = fmm_volatility_decay(np.linspace(0.0, 3.0, 13), 1.0, 2.0)
    assert g[:5].tolist() == [1.0] * 5 and g[8:].tolist() == [0.0] * 5
    assert g[6] == pytest.approx(0.5)
    assert fmm_volatility_decay(1.75, 1.0, 2.0) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="start < end"):
        fmm_volatility_decay(1.0, 2.0, 2.0)


def test_forward_looking_caplet_is_the_market_caplet():
    """With backward_looking=False the caplet is BGM's Black caplet on the same period."""
    cap = cap_floor_price(CURVE, 0.035, 2.0, 0.22, freq=4)
    fmm = fmm_caplet_price(CURVE, cap.reset_times, cap.payment_times, 0.035, 0.22, backward_looking=False)
    assert fmm == pytest.approx(cap.caplets, rel=1e-12)


@pytest.mark.parametrize("normal, vol", [(False, 0.25), (True, 0.01)])
def test_in_arrears_caplet_matches_simulation(normal, vol):
    times, R = simulate_fmm_rate(CURVE, 1.0, 1.5, vol, 30, 40_000, normal=normal, rng=0)
    tau, P = 0.5, float(CURVE.discount(1.5))
    for K, kind in ((0.03, "call"), (0.04, "put")):
        payoff = tau * P * np.maximum((1 if kind == "call" else -1) * (R[:, -1] - K), 0.0)
        exact = fmm_caplet_price(CURVE, 1.0, 1.5, K, vol, kind, normal=normal)
        assert abs(payoff.mean() - exact) < 4 * payoff.std(ddof=1) / np.sqrt(payoff.size)
    F = (CURVE.discount(1.0) / CURVE.discount(1.5) - 1) / tau
    assert np.all(R[:, 0] == pytest.approx(F))
    assert abs(R[:, -1].mean() - F) < 4 * R[:, -1].std() / np.sqrt(R.shape[0])  # a martingale
    # The rate keeps moving during the accrual period, but more and more slowly.
    moves = np.abs(np.diff(R, axis=1)).mean(axis=0)
    assert moves[-1] < 0.2 * moves[np.searchsorted(times, 1.0) - 1]


def test_backward_looking_variance_uplift():
    """The in-arrears caplet is Black with expiry start + tau/3, so it is worth more and has value even once the period has begun."""
    back = fmm_caplet_price(CURVE, [0.0, 1.0, 5.0], [0.25, 1.25, 5.25], 0.04, 0.3)
    fwd = fmm_caplet_price(CURVE, [0.0, 1.0, 5.0], [0.25, 1.25, 5.25], 0.04, 0.3, backward_looking=False)
    assert np.all(back > fwd) and back[0] > 0
    assert back[2] / fwd[2] < back[1] / fwd[1]  # the uplift fades with expiry
    floor = fmm_caplet_price(CURVE, 1.0, 1.25, 0.04, 0.3, "put", notional=1e6)
    cap = fmm_caplet_price(CURVE, 1.0, 1.25, 0.04, 0.3, "call", notional=1e6)
    P0, P1 = CURVE.discount(1.0), CURVE.discount(1.25)
    assert cap - floor == pytest.approx(1e6 * (P0 - P1 - 0.25 * 0.04 * P1), rel=1e-10)  # parity with the swaplet


def test_compounded_short_rate_has_the_curve_forward():
    """In a short-rate model the backward-looking rate is the compounded short rate, and its T_j-forward mean is R_j(0)."""
    model = HullWhiteModel(CURVE, 0.1, 0.01)
    times, r = model.simulate(1.5, 300, 20_000, rng=5)
    D = path_discount_factors(times, r)
    i = np.searchsorted(times, 1.0)
    compounded = (D[:, i] / D[:, -1] - 1.0) / 0.5
    value = D[:, -1] * 0.5 * compounded
    exact = float(CURVE.discount(1.0) - CURVE.discount(1.5))
    assert abs(value.mean() - exact) < 4 * value.std(ddof=1) / np.sqrt(value.size) + 1e-6


def test_fmm_validation():
    with pytest.raises(ValueError, match="vol must be non-negative"):
        fmm_caplet_price(CURVE, 1.0, 1.25, 0.04, -0.1)
    with pytest.raises(ValueError, match="positive forward"):
        simulate_fmm_rate(-0.01, 1.0, 1.25, 0.2, 10, 10, rng=0)
    t, R = simulate_fmm_rate(-0.01, 1.0, 1.25, 0.002, 10, 10, normal=True, rng=0)
    assert R.shape == (10, 11) and np.all(R[:, 0] < 0)
