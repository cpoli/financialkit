"""Tests for the Castagna-Mercurio vanna-volga smile."""

import numpy as np
import pytest

from financialkit.options import black_scholes_greeks, fx_pillar_strikes, garman_kohlhagen_price, vanna_volga_price, vanna_volga_smile

S, T, R_D, R_F = 1.20, 0.5, 0.03, 0.01
VOLS = np.array([0.106, 0.100, 0.099])
PILLARS = fx_pillar_strikes(S, T, R_D, R_F, *VOLS)


def test_pillar_strikes_have_the_quoted_deltas():
    """Castagna and Mercurio (2007), eqs. (2)-(4): spot deltas of -25%, 0 for the straddle, and +25%."""
    put = black_scholes_greeks(S, PILLARS[0], T, R_D, VOLS[0], R_F, "put").delta
    call = black_scholes_greeks(S, PILLARS[2], T, R_D, VOLS[2], R_F).delta
    straddle = black_scholes_greeks(S, PILLARS[1], T, R_D, VOLS[1], R_F).delta + black_scholes_greeks(S, PILLARS[1], T, R_D, VOLS[1], R_F, "put").delta
    assert put == pytest.approx(-0.25) and call == pytest.approx(0.25) and straddle == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_reproduces_the_pillar_quotes(option_type):
    price = vanna_volga_price(PILLARS, S, T, R_D, R_F, PILLARS, VOLS, option_type)
    assert price == pytest.approx(garman_kohlhagen_price(S, PILLARS, T, R_D, R_F, VOLS, option_type), rel=1e-12)


def test_flat_pillars_give_garman_kohlhagen():
    K = np.linspace(1.0, 1.4, 9)
    price = vanna_volga_price(K, S, T, R_D, R_F, PILLARS, [0.1, 0.1, 0.1])
    assert price == pytest.approx(garman_kohlhagen_price(S, K, T, R_D, R_F, 0.1), rel=1e-13)


def test_vanna_volga_satisfies_put_call_parity():
    K = np.linspace(1.0, 1.4, 9)
    c, p = vanna_volga_price(K, S, T, R_D, R_F, PILLARS, VOLS), vanna_volga_price(K, S, T, R_D, R_F, PILLARS, VOLS, "put")
    assert c - p == pytest.approx(S * np.exp(-R_F * T) - K * np.exp(-R_D * T), abs=1e-13)


def test_second_order_approximation_tracks_the_exact_smile():
    """Castagna and Mercurio (2007), Figure 1: the second-order formula is accurate well into the wings, the first is not."""
    K = np.linspace(1.05, 1.40, 15)
    exact = vanna_volga_smile(K, S, T, R_D, R_F, PILLARS, VOLS).vols
    second = vanna_volga_smile(K, S, T, R_D, R_F, PILLARS, VOLS, "second_order").vols
    first = vanna_volga_smile(K, S, T, R_D, R_F, PILLARS, VOLS, "first_order").vols
    inner = (K >= PILLARS[0]) & (K <= PILLARS[2])
    assert np.max(np.abs(second - exact)) < 1e-3
    assert np.max(np.abs(first - exact)[inner]) < 2e-4
    assert np.max(np.abs(first - exact)) > 5 * np.max(np.abs(second - exact))


def test_smile_has_the_quoted_skew_and_bottoms_out_near_the_money():
    K = np.linspace(1.05, 1.40, 71)
    vols = vanna_volga_smile(K, S, T, R_D, R_F, PILLARS, VOLS).vols
    assert vols[0] > vols[-1]  # risk reversal: the 25-delta put is dearer than the call
    assert PILLARS[1] < K[np.argmin(vols)] < PILLARS[2]
    assert vols.min() < VOLS.min()


@pytest.mark.parametrize(
    "pillars,vols,match",
    [([1.1, 1.2], [0.1, 0.1, 0.1], "three pillar"), ([1.3, 1.2, 1.1], [0.1, 0.1, 0.1], "increasing"), ([1.1, 1.2, 1.3], [0.1, -0.1, 0.1], "must be positive")],
)
def test_invalid_pillars_raise(pillars, vols, match):
    with pytest.raises(ValueError, match=match):
        vanna_volga_price(1.2, S, T, R_D, R_F, pillars, vols)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        vanna_volga_smile([1.2], S, T, R_D, R_F, PILLARS, VOLS, "third_order")
