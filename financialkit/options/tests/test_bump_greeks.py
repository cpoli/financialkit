"""Tests for financialkit.options.utils.common.bump_greek against the analytic Black-Scholes Greeks."""

import dataclasses

import numpy as np
import pytest

from financialkit.options import black_scholes_greeks, black_scholes_price, bump_greek

ARGS = (90.0, 100.0, 0.75, 0.04, 0.3, 0.02)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bump_greek_matches_black_scholes_greeks(option_type):
    g = black_scholes_greeks(*ARGS, option_type=option_type)
    fd = {
        "delta": bump_greek(black_scholes_price, "S", *ARGS, option_type=option_type),
        "gamma": bump_greek(black_scholes_price, ("S", "S"), *ARGS, option_type=option_type, h=1e-2),
        "vega": bump_greek(black_scholes_price, "sigma", *ARGS, option_type=option_type),
        "theta": -bump_greek(black_scholes_price, "T", *ARGS, option_type=option_type),
        "rho": bump_greek(black_scholes_price, "r", *ARGS, option_type=option_type),
        "epsilon": bump_greek(black_scholes_price, "q", *ARGS, option_type=option_type),
        "vanna": bump_greek(black_scholes_price, ("S", "sigma"), *ARGS, option_type=option_type, h={"S": 1e-2, "sigma": 1e-4}),
        "volga": bump_greek(black_scholes_price, ("sigma", "sigma"), *ARGS, option_type=option_type),
    }
    for name, value in fd.items():
        # O(h^2) truncation plus rounding of order eps * V / h^2 for the second differences
        assert value == pytest.approx(getattr(g, name), rel=1e-6, abs=1e-6), name


def test_relative_bump_and_keyword_arguments():
    exact = black_scholes_greeks(*ARGS).delta
    assert bump_greek(black_scholes_price, "S", S=90.0, K=100.0, T=0.75, r=0.04, sigma=0.3, q=0.02, h=1e-5, relative=True) == pytest.approx(exact, rel=1e-8)


def test_array_prices_give_array_greeks():
    K = np.array([80.0, 100.0, 120.0])
    vega = bump_greek(black_scholes_price, "sigma", 100.0, K, 1.0, 0.05, 0.2)
    np.testing.assert_allclose(vega, black_scholes_greeks(100.0, K, 1.0, 0.05, 0.2).vega, rtol=1e-7)


def test_model_attribute_is_bumped_on_a_copy() -> None:
    @dataclasses.dataclass(frozen=True)
    class Flat:
        sigma: float

    class Mutable:
        def __init__(self, sigma: float) -> None:
            self.sigma = sigma

    def price(model, S, K, T, r):
        return black_scholes_price(S, K, T, r, model.sigma)

    exact = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2).vega
    for model in (Flat(0.2), Mutable(0.2)):
        assert bump_greek(price, "model.sigma", model, 100.0, 100.0, 1.0, 0.05) == pytest.approx(exact, rel=1e-7)
        assert model.sigma == 0.2


def test_common_random_numbers_for_an_rng_argument():
    seeds = []

    def noisy(S, rng=None):
        seeds.append(rng)
        return S**2 + np.random.default_rng(rng).standard_normal()

    assert bump_greek(noisy, "S", 3.0, rng=np.random.default_rng(1)) == pytest.approx(6.0)
    assert bump_greek(noisy, ("S", "S"), 3.0, h=1e-2) == pytest.approx(2.0)
    assert bump_greek(noisy, "S", 3.0, rng=5) == pytest.approx(6.0)
    assert len(set(seeds[:2])) == 1 and isinstance(seeds[0], int) and seeds[-1] == 5


def test_bad_arguments():
    with pytest.raises(ValueError, match="no parameter 'vol'"):
        bump_greek(black_scholes_price, "vol", *ARGS)
    with pytest.raises(ValueError, match="one name or two"):
        bump_greek(black_scholes_price, ("S", "S", "S"), *ARGS)
    with pytest.raises(ValueError, match="bump of 'q' is zero"):
        bump_greek(black_scholes_price, "q", 100.0, 100.0, 1.0, 0.05, 0.2, relative=True)
