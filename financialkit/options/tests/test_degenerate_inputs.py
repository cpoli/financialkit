"""Closed-form prices and Greeks at T = 0, sigma = 0, K = 0, negative rates and negative spots.

Each must return its limit or raise ValueError, never nan or a
RuntimeWarning: the discounted intrinsic value of the forward when
sigma sqrt(T) = 0, the discounted forward for a call struck at zero.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from financialkit.options import (
    bachelier_price,
    black76_price,
    black_d1_d2,
    black_scholes_greeks,
    black_scholes_price,
    garman_kohlhagen_price,
)

S0, Q = 100.0, 0.02
BASE = {"S": S0, "K": 95.0, "T": 1.0, "r": 0.05, "s": 0.2}
LIMITS = {"T=0": {"T": 0.0}, "sigma=0": {"s": 0.0}, "K=0": {"K": 0.0}, "r<0": {"r": -0.02}}
GREEKS = ("delta", "gamma", "vega", "theta", "rho", "epsilon", "vanna", "volga")

PRICERS: dict[str, Callable[..., Any]] = {}
for ot in ("call", "put"):
    PRICERS[f"black_scholes-{ot}"] = lambda S, K, T, r, s, ot=ot: black_scholes_price(S, K, T, r, s, Q, ot)
    PRICERS[f"black76-{ot}"] = lambda S, K, T, r, s, ot=ot: black76_price(S * np.exp((r - Q) * T), K, T, r, s, ot)
    PRICERS[f"garman_kohlhagen-{ot}"] = lambda S, K, T, r, s, ot=ot: garman_kohlhagen_price(S, K, T, r, Q, s, ot)
    PRICERS[f"bachelier-{ot}"] = lambda S, K, T, r, s, ot=ot: bachelier_price(S, K, T, r, 100.0 * s, Q, ot)
    PRICERS[f"greeks-{ot}"] = lambda S, K, T, r, s, ot=ot: [getattr(black_scholes_greeks(S, K, T, r, s, Q, ot), g) for g in GREEKS]


def _quiet(f, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return np.asarray(f(**kw), dtype=float)


def _intrinsic(S, K, T, r, phi):
    return np.exp(-r * T) * max(phi * (S * np.exp((r - Q) * T) - K), 0.0)


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("name", PRICERS)
def test_finite_without_warnings(name, limit):
    # BASE is off the money, spot and forward, so no Greek is infinite here
    assert np.all(np.isfinite(_quiet(PRICERS[name], **{**BASE, **LIMITS[limit]})))


@pytest.mark.parametrize("name", [n for n in PRICERS if not n.startswith("bachelier")])
def test_negative_spot_raises(name):
    with pytest.raises(ValueError, match="non-negative"):
        _quiet(PRICERS[name], **{**BASE, "S": -S0})


@pytest.mark.parametrize("ot", ["call", "put"])
def test_bachelier_accepts_negative_spots(ot):
    """The normal model allows negative prices, so parity must still hold there."""
    c, p = (_quiet(lambda S, K, T, r, s, o=o: bachelier_price(S, K, T, r, s, Q, o), **{**BASE, "S": -S0, "s": 20.0}) for o in ("call", "put"))
    assert c - p == pytest.approx(-S0 * np.exp(-Q) - 95.0 * np.exp(-0.05))


@pytest.mark.parametrize("r", [0.05, -0.02])
@pytest.mark.parametrize("limit", [{"T": 0.0}, {"s": 0.0}], ids=["T=0", "sigma=0"])
@pytest.mark.parametrize("K", [80.0, 120.0])
@pytest.mark.parametrize("ot, phi", [("call", 1), ("put", -1)])
@pytest.mark.parametrize("name", ["black_scholes", "black76", "garman_kohlhagen", "bachelier"])
def test_riskless_limit_is_discounted_intrinsic(name, ot, phi, K, limit, r):
    kw = {**BASE, "K": K, "r": r, **limit}
    assert float(_quiet(PRICERS[f"{name}-{ot}"], **kw)) == pytest.approx(_intrinsic(kw["S"], K, kw["T"], r, phi), abs=1e-12)


@pytest.mark.parametrize("name", [n for n in PRICERS if not n.startswith(("bachelier", "greeks"))])
def test_zero_strike_call_is_discounted_forward(name):
    value = float(_quiet(PRICERS[name], **{**BASE, "K": 0.0}))
    assert value == pytest.approx(S0 * np.exp(-Q) if name.endswith("call") else 0.0, abs=1e-12)


@pytest.mark.parametrize("limit", [{"K": 0.0}, {"S": 0.0}, {"T": 0.0}, {"s": 0.0}], ids=["K=0", "S=0", "T=0", "sigma=0"])
@pytest.mark.parametrize("ot", ["call", "put"])
def test_greeks_off_the_money_match_their_limits(ot, limit):
    """Where the option is surely exercised or surely not, the Greeks of e^{-rT} max(phi (F - K), 0)."""
    kw = {**BASE, "K": 90.0, **limit}
    g = black_scholes_greeks(kw["S"], kw["K"], kw["T"], kw["r"], kw["s"], Q, ot)
    if limit.get("T") == 0.0 or limit.get("s") == 0.0:
        bump = {**kw, "T": max(kw["T"], 1e-12), "s": max(kw["s"], 1e-3)} if limit.get("T") == 0.0 else {**kw, "s": 1e-3}
    else:
        bump = {**kw, "K": max(kw["K"], 1e-9), "S": max(kw["S"], 1e-9)}
    h = black_scholes_greeks(bump["S"], bump["K"], bump["T"], bump["r"], bump["s"], Q, ot)
    for name in GREEKS:
        assert getattr(g, name) == pytest.approx(getattr(h, name), abs=1e-6), name


@pytest.mark.parametrize("ot, phi", [("call", 1), ("put", -1)])
def test_greeks_at_the_money_at_expiry(ot, phi):
    """At the kink delta is the midpoint, gamma is +inf and theta -inf: limits, not nan."""
    g = _quiet(PRICERS[f"greeks-{ot}"], **{**BASE, "K": S0, "T": 0.0})
    delta, gamma, vega, theta, rho, epsilon, vanna, volga = g
    assert delta == pytest.approx(0.5 * phi)
    assert gamma == np.inf and theta == -np.inf
    assert vega == rho == epsilon == vanna == volga == 0.0


@pytest.mark.parametrize("ot", ["call", "put"])
def test_greeks_at_the_money_forward_with_zero_volatility(ot):
    """sigma -> 0 with F = K: vanna tends to e^{-qT} n(0) sqrt(T) / 2, the rest are finite but gamma."""
    T, r = 1.0, 0.05
    K = S0 * np.exp((r - Q) * T)
    g = black_scholes_greeks(S0, K, T, r, 0.0, Q, ot)
    h = black_scholes_greeks(S0, K, T, r, 1e-9, Q, ot)
    assert g.gamma == np.inf
    assert g.vanna == pytest.approx(0.5 * np.exp(-Q * T) * norm.pdf(0.0) * np.sqrt(T))
    for name in ("delta", "vega", "theta", "rho", "epsilon", "vanna", "volga"):
        assert getattr(g, name) == pytest.approx(getattr(h, name), abs=1e-6), name


def test_black_d1_d2_limits():
    d1, d2 = black_d1_d2([110.0, 100.0, 90.0, 0.0, 50.0], [100.0, 100.0, 100.0, 100.0, 0.0], 0.2, 0.0)
    assert d1.tolist() == d2.tolist() == [np.inf, 0.0, -np.inf, -np.inf, np.inf]


@pytest.mark.parametrize("bad", [{"T": -1.0}, {"s": -0.2}, {"K": -100.0}])
def test_negative_maturity_volatility_or_strike_raises(bad):
    with pytest.raises(ValueError):
        _quiet(PRICERS["black_scholes-call"], **{**BASE, **bad})
