"""Tests for implied volatility, the Brenner-Subrahmanyam approximation and the smile."""

import time

import numpy as np
import pytest

from financialkit.constants import DEFAULT_ATOL
from financialkit.options import (
    bachelier_implied_volatility,
    bachelier_price,
    black76_price,
    black_scholes_greeks,
    black_scholes_price,
    brenner_subrahmanyam_vol,
    implied_vol_smile,
    implied_volatility,
    no_arbitrage_bounds,
)


def test_hull_section_15_11():
    """Hull, 10th ed., Section 15.11: a 1.875 call (S = 21, K = 20, r = 10%, 3 months) implies 23.5%."""
    assert implied_volatility(1.875, 21.0, 20.0, 0.25, 0.10) == pytest.approx(0.235, abs=5e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_round_trip_across_strikes_and_vols(option_type):
    K = np.array([[70.0], [100.0], [140.0]])
    sigma = np.array([[0.1, 0.3, 1.2]])
    prices = black_scholes_price(100.0, K, 1.5, 0.04, sigma, 0.02, option_type)
    np.testing.assert_allclose(implied_volatility(prices, 100.0, K, 1.5, 0.04, 0.02, option_type), np.broadcast_to(sigma, (3, 3)), atol=1e-7)


def test_black76_implied_vol_via_forward_and_q_equals_r():
    price = black76_price(20.0, 22.0, 0.5, 0.06, 0.3)
    assert implied_volatility(price, 20.0, 22.0, 0.5, 0.06, q=0.06) == pytest.approx(0.3, abs=1e-8)


@pytest.mark.parametrize("price", [0.0, 2.0, 100.0, 150.0])
def test_prices_outside_merton_bounds_raise(price):
    # lower bound of this call is 100 - 98 e^{-0.05} ~ 6.78, upper bound 100
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(price, 100.0, 98.0, 1.0, 0.05)


def test_brenner_subrahmanyam_error_is_third_order_in_vol():
    for sigma, T in [(0.1, 0.25), (0.2, 1.0), (0.4, 1.0)]:
        F = 100.0 * np.exp(0.03 * T)
        c = black_scholes_price(100.0, F, T, 0.03, sigma)
        approx = brenner_subrahmanyam_vol(c, 100.0, T, 0.03)
        # 2N(x/2) - 1 = x/sqrt(2 pi) (1 - x^2/24 + ...), so the relative error is ~ sigma^2 T / 24
        assert approx / sigma - 1 == pytest.approx(-(sigma**2) * T / 24, rel=0.1, abs=1e-6)


def test_smile_from_skewed_prices_slopes_down():
    K = np.linspace(70, 130, 13)
    # a mixture of two lognormals with the same forward: a standard arbitrage-free skew
    w, s1, s2 = 0.3, 0.45, 0.15
    prices = w * black_scholes_price(100.0, K, 0.5, 0.0, s1) + (1 - w) * black_scholes_price(100.0, K, 0.5, 0.0, s2)
    smile = implied_vol_smile(prices, K, 100.0, 0.5, 0.0)
    assert smile.vols.shape == K.shape and smile.forward == pytest.approx(100.0)
    # a symmetric mixture gives a smile: the wings exceed the at-the-money vol
    atm = smile.vols[K == 100.0][0]
    assert smile.vols[0] > atm and smile.vols[-1] > atm
    np.testing.assert_allclose(smile.log_moneyness, np.log(K / 100.0))


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize("T", [0.0, -0.5, np.array([1.0, 0.0])])
def test_non_positive_maturity_raises(T):
    # regression: at T = 0 the bracket doubling never ended
    with pytest.raises(ValueError, match="must be positive"):
        _timed(implied_volatility, 5.0, 100.0, 98.0, T, 0.05)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("bound", [0, 1])
def test_prices_at_the_bounds_raise(option_type, bound):
    price = no_arbitrage_bounds(100.0, 98.0, 1.0, 0.05, option_type=option_type)[bound]
    with pytest.raises(ValueError, match="no-arbitrage"):
        _timed(implied_volatility, price, 100.0, 98.0, 1.0, 0.05, option_type=option_type)


def test_price_just_above_the_lower_bound_raises():
    lower, _ = no_arbitrage_bounds(100.0, 98.0, 1.0, 0.05)
    with pytest.raises(ValueError, match="too close"):
        _timed(implied_volatility, np.nextafter(lower, np.inf), 100.0, 98.0, 1.0, 0.05)


def test_volatility_above_the_bracket_cap_raises():
    # with T = 1e-12, a call halfway to its upper bound needs sigma far beyond 1e4
    with pytest.raises(ValueError, match="too close"):
        _timed(implied_volatility, 50.0, 100.0, 98.0, 1e-12, 0.05)


def test_price_just_below_the_upper_bound_solves():
    _, upper = no_arbitrage_bounds(100.0, 98.0, 1.0, 0.05)
    sigma = _timed(implied_volatility, np.nextafter(upper, 0.0), 100.0, 98.0, 1.0, 0.05)
    assert 1.0 < sigma < 1e4


@pytest.mark.parametrize(("K", "option_type"), [(300.0, "call"), (20.0, "put")])
@pytest.mark.parametrize("sigma", [0.2, 0.3])
def test_deep_out_of_the_money_round_trip(K, option_type, sigma):
    price = black_scholes_price(100.0, K, 0.25, 0.05, sigma, option_type=option_type)
    assert price < 1e-10
    assert _timed(implied_volatility, price, 100.0, K, 0.25, 0.05, option_type=option_type) == pytest.approx(sigma, abs=1e-8)


def test_newton_matches_brentq_on_a_surface_and_is_faster():
    # C9: a 100 x 100 surface inverts in milliseconds and matches the brentq loop to DEFAULT_ATOL
    rng = np.random.default_rng(9)
    K, T = np.linspace(70.0, 140.0, 100)[:, None], np.linspace(0.25, 3.0, 100)[None, :]
    sigma = 0.2 + 0.3 * rng.random((100, 100))
    prices = black_scholes_price(100.0, K, T, 0.03, sigma, 0.01, "put")
    start = time.perf_counter()
    newton = implied_volatility(prices, 100.0, K, T, 0.03, 0.01, "put")
    t_newton = time.perf_counter() - start
    start = time.perf_counter()
    reference = implied_volatility(prices, 100.0, K, T, 0.03, 0.01, "put", method="brentq")
    t_brentq = time.perf_counter() - start
    np.testing.assert_allclose(newton, reference, rtol=0, atol=DEFAULT_ATOL)
    np.testing.assert_allclose(newton, sigma, rtol=0, atol=DEFAULT_ATOL)
    assert t_newton < t_brentq / 10


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_newton_round_trips_across_moneyness_maturity_and_vol(option_type):
    K = 100.0 * np.exp(np.linspace(-1.0, 1.0, 41))[:, None, None]
    T = np.array([1 / 365, 0.1, 1.0, 10.0])[None, :, None]
    sigma = np.array([0.05, 0.3, 1.0, 3.0])[None, None, :]
    prices = black_scholes_price(100.0, K, T, 0.02, sigma, option_type=option_type)
    vega = black_scholes_greeks(100.0, K, T, 0.02, sigma, option_type=option_type).vega
    # only where an ulp of the price moves sigma by less than 1e-12 is the volatility determined
    lower, upper = no_arbitrage_bounds(100.0, K, T, 0.02, option_type=option_type)
    resolvable = (prices > lower) & (prices < upper) & (np.spacing(prices) / np.maximum(vega, 1e-300) < 1e-13)
    vols = implied_volatility(
        prices[resolvable], 100.0, np.broadcast_to(K, prices.shape)[resolvable], np.broadcast_to(T, prices.shape)[resolvable], 0.02, option_type=option_type
    )
    np.testing.assert_allclose(vols, np.broadcast_to(sigma, prices.shape)[resolvable], rtol=0, atol=DEFAULT_ATOL)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        implied_volatility(10.0, 100.0, 100.0, 1.0, 0.05, method="secant")


def test_newton_iteration_cap_raises():
    with pytest.raises(RuntimeError, match="did not converge in 1 iterations"):
        implied_volatility(10.0, 100.0, 100.0, 1.0, 0.05, max_iter=1)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bachelier_implied_volatility_round_trips_negative_prices(option_type):
    """WTI crude oil futures settled at -37.63 on 20 April 2020; the normal model still inverts."""
    F = np.array([-37.63, -10.0, 0.0, 5.0, 20.0])
    K = np.array([-20.0, 0.0, 1.0, -5.0, 25.0])
    sigma_n = np.array([40.0, 15.0, 3.0, 8.0, 12.0])
    price = bachelier_price(F, K, 0.08, 0.01, sigma_n, q=0.01, option_type=option_type)
    assert bachelier_implied_volatility(price, F, K, 0.08, 0.01, 0.01, option_type) == pytest.approx(sigma_n, rel=1e-10)


def test_bachelier_implied_volatility_rejects_prices_at_intrinsic():
    with pytest.raises(ValueError, match="discounted intrinsic value"):
        bachelier_implied_volatility(5.0, 5.0, 0.0, 0.5, 0.0)
