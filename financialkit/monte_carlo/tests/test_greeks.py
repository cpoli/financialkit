"""Tests for financialkit.monte_carlo.systems.greeks against Black-Scholes Greeks."""

import numpy as np
import pytest
from scipy.stats import norm

from financialkit.monte_carlo import likelihood_ratio_greeks, pathwise_greeks
from financialkit.options import black_scholes_greeks, black_scholes_price

ARGS = (100.0, 105.0, 0.75, 0.04, 0.25)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_likelihood_ratio_greeks(option_type):
    exact = black_scholes_greeks(*ARGS, q=0.01, option_type=option_type)
    g = likelihood_ratio_greeks(*ARGS, q=0.01, option_type=option_type, n_paths=400_000, rng=1)
    assert abs(g.price - black_scholes_price(*ARGS, q=0.01, option_type=option_type)) < 4 * g.price_se
    assert abs(g.delta - exact.delta) < 4 * g.delta_se
    assert abs(g.vega - exact.vega) < 4 * g.vega_se
    assert abs(g.gamma - exact.gamma) < 4 * g.gamma_se


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_pathwise_greeks(option_type):
    exact = black_scholes_greeks(*ARGS, q=0.01, option_type=option_type)
    g = pathwise_greeks(*ARGS, q=0.01, option_type=option_type, n_paths=200_000, rng=2)
    assert abs(g.delta - exact.delta) < 4 * g.delta_se
    assert abs(g.vega - exact.vega) < 4 * g.vega_se
    assert np.isnan(g.gamma) and g.method == "pathwise"


def test_pathwise_has_lower_variance_on_common_numbers():
    z = np.random.default_rng(3).standard_normal(100_000)
    lr, pw = likelihood_ratio_greeks(*ARGS, z=z), pathwise_greeks(*ARGS, z=z)
    assert lr.price == pytest.approx(pw.price)
    assert pw.delta_se < lr.delta_se and pw.vega_se < lr.vega_se


def test_likelihood_ratio_digital_delta():
    """Cash-or-nothing call: V = e^{-rT} N(d2), delta = e^{-rT} n(d2) / (S sigma sqrt T)."""
    S, K, T, r, sig = ARGS
    d2 = (np.log(S / K) + (r - 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    g = likelihood_ratio_greeks(*ARGS, digital=True, n_paths=400_000, rng=4)
    assert abs(g.price - np.exp(-r * T) * norm.cdf(d2)) < 4 * g.price_se
    assert abs(g.delta - np.exp(-r * T) * norm.pdf(d2) / (S * sig * np.sqrt(T))) < 4 * g.delta_se
