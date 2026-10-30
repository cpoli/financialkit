"""Every exotic closed form at T = 0, sigma = 0, K = 0, negative rates and negative spots.

Each must return its limit or raise ValueError, never nan or a
RuntimeWarning. With no volatility, or no time left, the price must be the
discounted payoff of the one riskless path S e^{bt}, which these tests
evaluate on a fine grid, independently of the closed forms.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.integrate import trapezoid

from financialkit.exotic_options import (
    asset_or_nothing_price,
    barrier_price,
    bivariate_normal_cdf,
    cash_or_nothing_price,
    chooser_price,
    compound_option_price,
    discrete_barrier_price,
    double_barrier_price,
    fixed_lookback_price,
    floating_lookback_price,
    forward_start_price,
    gap_option_price,
    geometric_asian_price,
    margrabe_price,
    merton_down_and_out_call,
    one_touch_price,
    quanto_price,
    stulz_price,
    turnbull_wakeman_price,
)
from financialkit.options import black_scholes_price

S0, Q, R_F, N_GRID = 100.0, 0.02, 0.03, 240_001
DOWN, UP = 90.0, 105.0  # the riskless path crosses DOWN when r < q and UP when r > q, by T = 3


def _path(S, T, r):
    t = np.linspace(0.0, T, N_GRID)
    return t, S * np.exp((r - Q) * t)


def _vanilla(x, K, phi):
    return np.maximum(phi * (x - K), 0.0)


def _barrier_pv(S, K, T, r, H, phi, barrier_type, rebate):
    t, p = _path(S, T, r)
    down, out = barrier_type.startswith("down"), barrier_type.endswith("out")
    crossed = p <= H if down else p >= H
    hit = crossed.any()
    t_hit = t[np.argmax(crossed)]
    vanilla = np.exp(-r * T) * _vanilla(p[-1], K, phi)
    if out:
        return rebate * np.exp(-r * t_hit) if hit else vanilla
    return vanilla if hit else rebate * np.exp(-r * T)


def _mean(t, x):
    return x[0] if t[-1] == 0 else trapezoid(x, t) / t[-1]


def _fixings(t, p, n=12):
    return p[(N_GRID - 1) // n :: (N_GRID - 1) // n]


# name -> (pricer(S, K, T, r, sigma), riskless value(S, K, T, r, sigma))
CASES: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {}
for ot, phi in (("call", 1), ("put", -1)):
    CASES[f"cash_or_nothing-{ot}"] = (
        lambda S, K, T, r, s, ot=ot: cash_or_nothing_price(S, K, T, r, s, Q, ot),
        lambda S, K, T, r, s, phi=phi: np.exp(-r * T) * float(phi * (_path(S, T, r)[1][-1] - K) > 0),
    )
    CASES[f"asset_or_nothing-{ot}"] = (
        lambda S, K, T, r, s, ot=ot: asset_or_nothing_price(S, K, T, r, s, Q, ot),
        lambda S, K, T, r, s, phi=phi: np.exp(-r * T) * _path(S, T, r)[1][-1] * float(phi * (_path(S, T, r)[1][-1] - K) > 0),
    )
    CASES[f"gap-{ot}"] = (
        lambda S, K, T, r, s, ot=ot: gap_option_price(S, K, 1.05 * K, T, r, s, Q, ot),
        lambda S, K, T, r, s, phi=phi: np.exp(-r * T) * phi * (_path(S, T, r)[1][-1] - 1.05 * K) * float(phi * (_path(S, T, r)[1][-1] - K) > 0),
    )
    CASES[f"geometric_asian-{ot}"] = (
        lambda S, K, T, r, s, ot=ot: geometric_asian_price(S, K, T, r, s, Q, ot),
        lambda S, K, T, r, s, phi=phi: np.exp(-r * T) * _vanilla(np.exp(_mean(*(lambda t, p: (t, np.log(p)))(*_path(S, T, r)))), K, phi),
    )
    CASES[f"geometric_asian_12-{ot}"] = (
        lambda S, K, T, r, s, ot=ot: geometric_asian_price(S, K, T, r, s, Q, ot, n_fixings=12),
        lambda S, K, T, r, s, phi=phi: np.exp(-r * T) * _vanilla(np.exp(np.mean(np.log(_fixings(*_path(S, T, r))))), K, phi),
    )
    CASES[f"turnbull_wakeman-{ot}"] = (
        lambda S, K, T, r, s, ot=ot: turnbull_wakeman_price(S, K, T, r, s, Q, ot),
        lambda S, K, T, r, s, phi=phi: np.exp(-r * T) * _vanilla(_mean(*_path(S, T, r)), K, phi),
    )
    CASES[f"turnbull_wakeman_12-{ot}"] = (
        lambda S, K, T, r, s, ot=ot: turnbull_wakeman_price(S, K, T, r, s, Q, ot, n_fixings=12),
        lambda S, K, T, r, s, phi=phi: np.exp(-r * T) * _vanilla(np.mean(_fixings(*_path(S, T, r))), K, phi),
    )
    CASES[f"floating_lookback-{ot}"] = (
        lambda S, K, T, r, s, ot=ot: floating_lookback_price(S, T, r, s, Q, ot),
        lambda S, K, T, r, s, phi=phi: np.exp(-r * T) * phi * (_path(S, T, r)[1][-1] - (_path(S, T, r)[1].min() if phi == 1 else _path(S, T, r)[1].max())),
    )
    CASES[f"fixed_lookback-{ot}"] = (
        lambda S, K, T, r, s, ot=ot: fixed_lookback_price(S, K, T, r, s, Q, ot),
        lambda S, K, T, r, s, phi=phi: np.exp(-r * T) * _vanilla(_path(S, T, r)[1].max() if phi == 1 else _path(S, T, r)[1].min(), K, phi),
    )
    CASES[f"forward_start-{ot}"] = (
        lambda S, K, T, r, s, ot=ot: forward_start_price(S, K / S0, T / 4, T, r, s, Q, ot),
        lambda S, K, T, r, s, phi=phi: np.exp(-r * T) * _vanilla(_path(S, T, r)[1][-1], K / S0 * _path(S, T, r)[1][(N_GRID - 1) // 4], phi),
    )
    CASES[f"quanto-{ot}"] = (
        lambda S, K, T, r, s, ot=ot: quanto_price(S, K, T, r, R_F, s, 0.1, 0.3, Q, 1.5, ot),
        lambda S, K, T, r, s, phi=phi: 1.5 * np.exp(-r * T) * _vanilla(S * np.exp((R_F - Q) * T), K, phi),
    )
    for kind, ext in (("max", np.maximum), ("min", np.minimum)):
        CASES[f"stulz_{kind}-{ot}"] = (
            lambda S, K, T, r, s, ot=ot, kind=kind: stulz_price(S, 1.05 * S, K, T, r, s, 1.2 * s, 0.5, Q, Q, ot, kind),
            lambda S, K, T, r, s, phi=phi, ext=ext: np.exp(-r * T) * _vanilla(ext(1.0, 1.05) * S * np.exp((r - Q) * T), K, phi),
        )
    for barrier_type in ("down-and-out", "down-and-in", "up-and-out", "up-and-in"):
        H = DOWN if barrier_type.startswith("down") else UP
        CASES[f"barrier_{barrier_type}-{ot}"] = (
            lambda S, K, T, r, s, ot=ot, bt=barrier_type, H=H: barrier_price(S, K, H, T, r, s, Q, ot, bt, rebate=2.0),
            lambda S, K, T, r, s, phi=phi, bt=barrier_type, H=H: _barrier_pv(S, K, T, r, H, phi, bt, 2.0),
        )
        CASES[f"discrete_barrier_{barrier_type}-{ot}"] = (
            lambda S, K, T, r, s, ot=ot, bt=barrier_type, H=H: discrete_barrier_price(S, K, H, T, r, s, Q, ot, bt, n_monitor=50, rebate=2.0),
            lambda S, K, T, r, s, phi=phi, bt=barrier_type, H=H: _barrier_pv(S, K, T, r, H, phi, bt, 2.0),
        )
    for knock in ("out", "in"):
        CASES[f"double_barrier_{knock}-{ot}"] = (
            lambda S, K, T, r, s, ot=ot, knock=knock: double_barrier_price(S, K, DOWN, UP, T, r, s, Q, ot, knock),
            lambda S, K, T, r, s, phi=phi, knock=knock: (
                _barrier_pv(S, K, T, r, DOWN, phi, f"down-and-{knock}", 0.0) if (r - Q) < 0 else _barrier_pv(S, K, T, r, UP, phi, f"up-and-{knock}", 0.0)
            ),
        )
    for ut in ("call", "put"):
        K1 = 5.0 if ut == "call" else 3.0
        CASES[f"compound_{ot}_on_{ut}"] = (
            lambda S, K, T, r, s, ot=ot, ut=ut, K1=K1: compound_option_price(S, K1, K, T, T + 0.5, r, s, Q, ot, ut),
            lambda S, K, T, r, s, phi=phi, ut=ut, K1=K1: np.exp(-r * T) * _vanilla(black_scholes_price(_path(S, T, r)[1][-1], K, 0.5, r, s, Q, ut), K1, phi),
        )
CASES["merton_down_and_out"] = (
    lambda S, K, T, r, s: merton_down_and_out_call(S, K, DOWN, T, r, s, Q),
    lambda S, K, T, r, s: _barrier_pv(S, K, T, r, DOWN, 1, "down-and-out", 0.0),
)
CASES["chooser"] = (
    lambda S, K, T, r, s: chooser_price(S, K, T / 2, T, r, s, Q),
    lambda S, K, T, r, s: np.exp(-r * T) * abs(_path(S, T, r)[1][-1] - K),
)
CASES["margrabe"] = (
    lambda S, K, T, r, s: margrabe_price(S, K, T, s, 0.5 * s, 0.3, Q, 0.0),
    lambda S, K, T, r, s: max(S * np.exp(-Q * T) - K, 0.0),
)
for H in (DOWN, UP):
    CASES[f"one_touch_{H:g}"] = (
        lambda S, K, T, r, s, H=H: one_touch_price(S, H, T, r, s, Q),
        # a knock-out put struck below zero is worthless unless knocked out, when it pays its rebate of 1
        lambda S, K, T, r, s, H=H: _barrier_pv(S, -1.0, T, r, H, -1, ("down" if H < S else "up") + "-and-out", 1.0),
    )

LIMITS = {"T=0": {"T": 0.0}, "sigma=0": {"sigma": 0.0}, "K=0": {"K": 0.0}, "r<0": {"r": -0.02}}
BASE = {"S": S0, "K": 100.0, "T": 1.0, "r": 0.05, "s": 0.2}
# A zero underlying strike leaves Geske's critical price undefined, and Merton's formula needs H <= K.
RAISES = {(name, "K=0") for name in CASES if name.startswith("compound") or name == "merton_down_and_out"}


def _quiet(f, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return np.asarray(f(**kw), dtype=float)


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("name", CASES)
def test_finite_without_warnings(name, limit):
    kw = {**BASE, **{("s" if k == "sigma" else k): v for k, v in LIMITS[limit].items()}}
    if (name, limit) in RAISES:
        with pytest.raises(ValueError):
            _quiet(CASES[name][0], **kw)
    else:
        assert np.all(np.isfinite(_quiet(CASES[name][0], **kw)))


@pytest.mark.parametrize("name", CASES)
def test_negative_spot_raises(name):
    with pytest.raises(ValueError):
        _quiet(CASES[name][0], **{**BASE, "S": -S0})


@pytest.mark.parametrize("r", [0.05, -0.02], ids=["r>q", "r<q"])
@pytest.mark.parametrize("limit", [{"T": 0.0}, {"T": 3.0, "s": 0.0}], ids=["T=0", "sigma=0"])
@pytest.mark.parametrize("name", CASES)
def test_riskless_limit_is_the_path_payoff(name, limit, r):
    pricer, riskless = CASES[name]
    kw = {**BASE, "K": 95.0, "r": r, **limit}
    assert float(_quiet(pricer, **kw)) == pytest.approx(float(riskless(**kw)), rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("name", [name for name in CASES if (name, "K=0") not in RAISES])
def test_zero_strike_is_the_limit(name):
    pricer = CASES[name][0]
    assert float(_quiet(pricer, **{**BASE, "K": 0.0})) == pytest.approx(float(_quiet(pricer, **{**BASE, "K": 1e-9})), abs=1e-6)


def test_bivariate_normal_cdf_infinite_limits():
    from scipy.stats import norm

    h = np.array([np.inf, -np.inf, 0.3, 0.3, np.inf, -np.inf])
    k = np.array([0.3, 0.3, np.inf, -np.inf, np.inf, np.inf])
    expected = [norm.cdf(0.3), 0.0, norm.cdf(0.3), 0.0, 1.0, 0.0]
    assert np.allclose(bivariate_normal_cdf(h, k, 0.4), expected, atol=1e-15)


@pytest.mark.parametrize("sigma", [0.2, 0.0])
def test_gap_put_call_parity(sigma):
    """Regression: the gap put pays K2 - S_T, so c - p = S e^{-qT} - K2 e^{-rT}."""
    S, K1, K2, T, r = 50.0, 50.0, 57.0, 0.5, 0.09
    c, p = (gap_option_price(S, K1, K2, T, r, sigma, Q, ot) for ot in ("call", "put"))
    assert c - p == pytest.approx(S * np.exp(-Q * T) - K2 * np.exp(-r * T), abs=1e-12)


def test_stulz_with_one_riskless_asset_raises():
    with pytest.raises(ValueError, match="sigma1, sigma2"):
        stulz_price(100.0, 105.0, 100.0, 1.0, 0.05, 0.0, 0.2, 0.5)


def test_compound_zero_strike_of_underlying_raises():
    with pytest.raises(ValueError, match="K2"):
        compound_option_price(100.0, 5.0, 0.0, 0.5, 1.0, 0.05, 0.2)
