"""Tests for financialkit.exotic_options.systems.binaries."""

import numpy as np
import pytest
from scipy.stats import norm

from financialkit.constants import DEFAULT_RTOL
from financialkit.exotic_options import asset_or_nothing_price, barrier_price, cash_or_nothing_price, gap_option_price, one_touch_price
from financialkit.options import black_scholes_price


def test_haug_binary_examples():
    """Haug (2007), Sections 4.19.1-4.19.3."""
    assert cash_or_nothing_price(100.0, 80.0, 0.75, 0.06, 0.35, q=0.06, option_type="put", cash=10.0) == pytest.approx(2.6710, abs=1e-4)
    assert asset_or_nothing_price(70.0, 65.0, 0.5, 0.07, 0.27, q=0.05, option_type="put") == pytest.approx(20.2069, abs=1e-4)
    assert gap_option_price(50.0, 50.0, 57.0, 0.5, 0.09, 0.20) == pytest.approx(-0.0053, abs=1e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_is_asset_minus_cash_digitals(option_type):
    K = np.linspace(60.0, 140.0, 9)
    sign = 1 if option_type == "call" else -1
    combo = sign * (
        asset_or_nothing_price(100.0, K, 1.0, 0.04, 0.3, 0.01, option_type) - cash_or_nothing_price(100.0, K, 1.0, 0.04, 0.3, 0.01, option_type, cash=K)
    )
    np.testing.assert_allclose(combo, black_scholes_price(100.0, K, 1.0, 0.04, 0.3, 0.01, option_type), rtol=DEFAULT_RTOL)


def test_digital_call_plus_put_is_a_bond():
    c = cash_or_nothing_price(100.0, 95.0, 2.0, 0.03, 0.2)
    p = cash_or_nothing_price(100.0, 95.0, 2.0, 0.03, 0.2, option_type="put")
    assert c + p == pytest.approx(np.exp(-0.06))


@pytest.mark.parametrize("H", [80.0, 125.0])
def test_one_touch_with_zero_rate_is_the_hitting_probability(H):
    """With r = 0 the one-touch is P(hit before T), from the reflection principle for drifted Brownian motion."""
    S, T, sigma, q = 100.0, 1.5, 0.3, 0.02
    nu = -q - 0.5 * sigma**2
    x = np.log(H / S)
    s = sigma * np.sqrt(T)
    if H < S:
        p = norm.cdf((x - nu * T) / s) + np.exp(2 * nu * x / sigma**2) * norm.cdf((x + nu * T) / s)
    else:
        p = norm.cdf((-x + nu * T) / s) + np.exp(2 * nu * x / sigma**2) * norm.cdf((-x - nu * T) / s)
    assert one_touch_price(S, H, T, 0.0, sigma, q) == pytest.approx(p, rel=1e-10)


def test_one_touch_is_the_knock_out_rebate_and_tends_to_perpetual():
    rebate_part = barrier_price(100.0, 100.0, 90.0, 1.0, 0.05, 0.2, rebate=1.0) - barrier_price(100.0, 100.0, 90.0, 1.0, 0.05, 0.2)
    assert one_touch_price(100.0, 90.0, 1.0, 0.05, 0.2) == pytest.approx(rebate_part, rel=1e-12)
    mu = (0.05 - 0.02) / 0.04
    lam = np.sqrt(mu**2 + 2 * 0.05 / 0.04)
    assert one_touch_price(100.0, 90.0, 1e4, 0.05, 0.2) == pytest.approx(0.9 ** (mu + lam), rel=1e-8)
    assert one_touch_price(100.0, 110.0, 1e4, 0.05, 0.2) == pytest.approx(1.1 ** (mu - lam), rel=1e-8)
    assert one_touch_price(100.0, 100.0, 1.0, 0.05, 0.2, cash=5.0) == 5.0
