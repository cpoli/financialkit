"""Rate options and model bond prices at T = 0, sigma = 0, K = 0 and negative rates.

Each must return its limit or raise ValueError, never nan or a
RuntimeWarning. A bond option with no volatility or no time left is worth
its exercise value on today's curve; Black's formula, being lognormal,
must refuse the negative forwards that negative rates produce.
"""

import warnings

import numpy as np
import pytest

from financialkit.interest_rate_models import (
    G2PlusPlusModel,
    HoLeeModel,
    HullWhiteModel,
    MertonShortRateModel,
    VasicekModel,
    black_swaption_price,
    cap_floor_price,
    forward_swap_annuity,
    forward_swap_rate,
    jamshidian_swaption,
    model_cap_floor_price,
    zero_coupon_bond_option,
)

MODELS = {
    "merton": lambda r, s: MertonShortRateModel(r, 0.0, s),
    "vasicek": lambda r, s: VasicekModel(r, 0.2, 0.04, s),
    "ho_lee": lambda r, s: HoLeeModel(r, s),
    "hull_white": lambda r, s: HullWhiteModel(r, 0.1, s),
    "g2pp": lambda r, s: G2PlusPlusModel(r, 0.5, s, 0.05, s, -0.5),
}
RATES = {"r>0": 0.03, "r<0": -0.01}


def _quiet(f, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = f(*args, **kw)
    return float(getattr(out, "price", out))


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("model", MODELS)
def test_bond_prices_are_one_at_maturity(model, rate):
    m = MODELS[model](RATES[rate], 0.01)
    state = [0.0, 0.0] if model == "g2pp" else None
    assert _quiet(m.zero_coupon_bond, 0.0) == pytest.approx(1.0)
    assert _quiet(m.zero_coupon_bond, 2.0, t=2.0, state=state) == pytest.approx(1.0)


@pytest.mark.parametrize("ot, phi", [("call", 1), ("put", -1)])
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("limit", ["T=0", "sigma=0"])
@pytest.mark.parametrize("model", MODELS)
def test_bond_option_riskless_limit_is_exercise_value(model, limit, rate, ot, phi):
    """With sigma_P = 0 the option is P(0, T) max(phi (P(0, S)/P(0, T) - K), 0)."""
    m = MODELS[model](RATES[rate], 0.0 if limit == "sigma=0" else 0.01)
    T, S, K = (0.0 if limit == "T=0" else 1.0), 5.0, 0.9
    PT, PS = float(m.zero_coupon_bond(T)) if T > 0 else 1.0, float(m.zero_coupon_bond(S))
    assert _quiet(zero_coupon_bond_option, m, T, S, K, ot) == pytest.approx(max(phi * (PS - K * PT), 0.0), abs=1e-12)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("model", MODELS)
def test_bond_option_zero_strike(model, rate):
    m = MODELS[model](RATES[rate], 0.01)
    assert _quiet(zero_coupon_bond_option, m, 1.0, 5.0, 0.0, "call") == pytest.approx(float(m.zero_coupon_bond(5.0)))
    assert _quiet(zero_coupon_bond_option, m, 1.0, 5.0, 0.0, "put") == 0.0


@pytest.mark.parametrize("strike", [0.0, 0.03])
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("sigma", [0.0, 0.01])
@pytest.mark.parametrize("model", MODELS)
def test_model_caps_floors_and_swaptions_are_finite(model, sigma, rate, strike):
    """Cap minus floor is the forward-starting swap at every limit, negative rates included."""
    m = MODELS[model](RATES[rate], sigma)
    cap, floor = (model_cap_floor_price(m, strike, 3.0, kind=k) for k in ("cap", "floor"))
    swap = np.sum((cap.payment_times - cap.reset_times) * np.asarray(m.zero_coupon_bond(cap.payment_times)) * (cap.forwards - strike))
    assert _quiet(lambda: cap.price) - _quiet(lambda: floor.price) == pytest.approx(swap, abs=1e-12)
    if model != "g2pp":
        for expiry in (0.0, 1.0):
            assert np.isfinite(_quiet(jamshidian_swaption, m, expiry, 3.0, strike))


@pytest.mark.parametrize("payer, phi", [(True, 1), (False, -1)])
@pytest.mark.parametrize("limit", [{"vol": 0.0}, {"expiry": 0.0}, {"strike": 0.0}], ids=["vol=0", "T=0", "K=0"])
def test_black_swaption_limits(limit, payer, phi):
    kw = {"curve": 0.03, "expiry": 1.0, "tenor": 3.0, "strike": 0.03, "vol": 0.2, **limit}
    price = _quiet(black_swaption_price, payer=payer, **kw)
    A, F = forward_swap_annuity(0.03, kw["expiry"], 3.0), forward_swap_rate(0.03, kw["expiry"], 3.0)
    expected = (A * F if payer else 0.0) if "strike" in limit else A * max(phi * (F - kw["strike"]), 0.0)
    assert price == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("kind", ["cap", "floor"])
def test_black_cap_floor_zero_volatility_and_strike(kind):
    phi = 1 if kind == "cap" else -1
    flat = cap_floor_price(0.03, 0.03, 3.0, 0.0, kind=kind)
    tau = flat.payment_times - flat.reset_times
    discount = np.exp(-0.03 * flat.payment_times)
    assert _quiet(lambda: flat.price) == pytest.approx(np.sum(tau * discount * np.maximum(phi * (flat.forwards - 0.03), 0.0)), abs=1e-12)
    zero = cap_floor_price(0.03, 0.0, 3.0, 0.2, kind=kind)
    assert _quiet(lambda: zero.price) == pytest.approx(np.sum(tau * discount * zero.forwards) if kind == "cap" else 0.0, abs=1e-12)


@pytest.mark.parametrize(
    "price",
    [
        lambda: cap_floor_price(-0.01, 0.01, 3.0, 0.2),
        lambda: cap_floor_price(-0.01, -0.005, 3.0, 0.2, kind="floor"),
        lambda: black_swaption_price(-0.01, 1.0, 3.0, 0.01, 0.2),
    ],
    ids=["cap", "floor", "swaption"],
)
def test_black_rejects_negative_forwards(price):
    with pytest.raises(ValueError, match="bachelier_price"):
        price()
