"""Tests for Jäckel's "Let's be rational" implied volatility."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from financialkit.options import black_scholes_price, implied_volatility, normalised_black_call, normalised_implied_volatility
from financialkit.options.systems.lets_be_rational import _vega

EPS = np.finfo(np.float64).eps


def _reference(x, s):
    """b(x, s) = integral of the normalised vega from 0 to s: positive integrand, no cancellation."""
    # break at the inflection point sqrt(2|x|), where the integrand peaks
    peak = np.sqrt(2 * abs(x))
    return quad(lambda u: float(_vega(np.float64(x), np.float64(u))), 0.0, s, epsabs=0.0, epsrel=1e-13, limit=500, points=[peak] if peak < s else None)[0]


@pytest.mark.parametrize(
    "x,s",
    [
        (-30.0, 1.0),  # asymptotic expansion
        (-0.3, 0.05),  # small-t series
        (-0.05, 0.4),
        (-1.0, 2.0),  # erfcx
        (-0.01, 3.0),  # normal CDF
    ],
)
def test_normalised_black_is_accurate_in_every_regime(x, s):
    """Jäckel (2015), Section 3: each evaluation branch keeps full relative precision."""
    assert normalised_black_call(x, s) == pytest.approx(_reference(x, s), rel=1e-12)


def test_normalised_black_is_the_black_formula():
    F, K, T, sigma = 105.0, 100.0, 0.7, 0.25
    undiscounted = black_scholes_price(F, K, T, 0.0, sigma)
    assert normalised_black_call(np.log(F / K), sigma * np.sqrt(T)) * np.sqrt(F * K) == pytest.approx(undiscounted, rel=1e-14)


def test_at_the_money_is_closed_form():
    s = np.array([1e-6, 0.1, 1.0, 3.0])
    beta = 2 * norm.cdf(0.5 * s) - 1
    assert normalised_implied_volatility(beta, 0.0) == pytest.approx(s, rel=1e-9)


def test_two_iterations_reach_machine_precision():
    """Jäckel (2015), Section 5: two Householder steps from the rational guess are enough out of the money."""
    X, S = np.meshgrid(-np.logspace(-6, 1.3, 40), np.logspace(-2.5, 0.7, 40))
    beta = normalised_black_call(X, S)
    keep = beta > 1e-300  # the rest underflow, and no volatility is implied by a zero price
    X, S, beta = X[keep], S[keep], beta[keep]
    rel = np.abs(normalised_implied_volatility(beta, X) - S) / S
    assert np.median(rel) < 4 * EPS
    assert rel.max() < 1e-12  # the worst points have prices near 1e-280, where b itself is only accurate to ~1e-13


def test_guess_alone_is_already_close():
    X, S = np.meshgrid(-np.logspace(-3, 1, 20), np.logspace(-2, 0.5, 20))
    beta = normalised_black_call(X, S)
    keep = beta > 1e-300
    guess = normalised_implied_volatility(beta[keep], X[keep], n_iter=0)
    assert np.max(np.abs(guess - S[keep]) / S[keep]) < 0.1


def test_zero_time_value_and_upper_bound():
    out = normalised_implied_volatility(np.array([0.0, np.exp(-0.25)]), np.array([-0.5, -0.5]))
    assert out[0] == 0.0 and out[1] == np.inf


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_method_jaeckel_matches_newton(option_type):
    # out of the money only: deep in the money the time value is lost to rounding and neither method can recover 0.3
    K = (np.linspace(101.0, 160.0, 21) if option_type == "call" else np.linspace(60.0, 99.0, 21))[:, None]
    T = np.array([0.05, 0.5, 2.0, 10.0])
    prices = black_scholes_price(100.0, K, T, 0.03, 0.3, 0.01, option_type)
    jaeckel = implied_volatility(prices, 100.0, K, T, 0.03, 0.01, option_type, method="jaeckel")
    newton = implied_volatility(prices, 100.0, K, T, 0.03, 0.01, option_type)
    assert jaeckel == pytest.approx(newton, abs=1e-9)
    assert jaeckel == pytest.approx(0.3, abs=1e-10)


def test_jaeckel_inverts_vanishing_prices():
    """A 3-month option 20% out of the money is worth about 1e-75 at a 2% volatility and 3e-293 at 1%."""
    price = black_scholes_price(100.0, 120.0, 0.25, 0.0, np.array([0.02, 0.01]))
    assert 0 < price[1] < price[0] < 1e-70
    vols = implied_volatility(price, 100.0, 120.0, 0.25, 0.0, method="jaeckel")
    assert vols[0] == pytest.approx(0.02, rel=1e-13)
    assert vols[1] == pytest.approx(0.01, rel=1e-11)  # exp(-h^2 / 2) at h = -36 magnifies the rounding of h


def test_jaeckel_resolves_volatilities_beyond_the_bracket_cap():
    """The Newton and brentq bracket stops at sigma = 1e4; Jäckel's iteration has no bracket."""
    price = black_scholes_price(100.0, 100.0, 1e-8, 0.0, 2e4)
    with pytest.raises(ValueError, match="too close to the no-arbitrage bounds"):
        implied_volatility(price, 100.0, 100.0, 1e-8, 0.0)
    assert implied_volatility(price, 100.0, 100.0, 1e-8, 0.0, method="jaeckel") == pytest.approx(2e4, rel=1e-12)
