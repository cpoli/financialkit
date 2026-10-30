"""Tests for the Haug-Haug-Lewis discrete-dividend pricer."""

import numpy as np
import pytest

from financialkit.options import black_scholes_price, discrete_dividend_price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_dividend_at_expiry_shifts_the_strike(option_type):
    """A drop of D at T makes max(S_T - D - K, 0) a call struck at K + D (Haug, Haug and Lewis 2003)."""
    price = discrete_dividend_price(100.0, [90.0, 100.0, 110.0], 1.0, 0.05, 0.3, [1.0], [4.0], option_type)
    assert price == pytest.approx(black_scholes_price(100.0, np.array([94.0, 104.0, 114.0]), 1.0, 0.05, 0.3, option_type=option_type), abs=1e-10)


def test_dividend_now_reduces_the_spot():
    price = discrete_dividend_price(100.0, 100.0, 1.0, 0.05, 0.3, [1e-12], [4.0])
    assert price == pytest.approx(black_scholes_price(96.0, 100.0, 1.0, 0.05, 0.3), abs=1e-10)


def test_no_dividend_is_black_scholes():
    price = discrete_dividend_price(100.0, 100.0, 1.0, 0.05, 0.3, [0.25, 0.5], [0.0, 0.0], "put")
    assert price == pytest.approx(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.3, option_type="put"), abs=1e-10)


def test_dividends_after_expiry_are_ignored():
    assert discrete_dividend_price(100.0, 100.0, 0.5, 0.05, 0.3, [0.75], [5.0]) == pytest.approx(black_scholes_price(100.0, 100.0, 0.5, 0.05, 0.3))


def test_three_dividends_match_monte_carlo():
    """GBM between dates and a drop of D (floored at zero) on each date, simulated exactly."""
    rng = np.random.default_rng(7)
    n, r, sigma = 400_000, 0.05, 0.3
    S, t = np.full(n, 100.0), 0.0
    for t_i in (0.25, 0.5, 0.75):
        S = np.maximum(S * np.exp((r - 0.5 * sigma**2) * (t_i - t) + sigma * np.sqrt(t_i - t) * rng.standard_normal(n)) - 2.0, 0.0)
        t = t_i
    S = S * np.exp((r - 0.5 * sigma**2) * (1 - t) + sigma * np.sqrt(1 - t) * rng.standard_normal(n))
    payoff = np.exp(-r) * np.maximum(100.0 - S, 0.0)
    exact = discrete_dividend_price(100.0, 100.0, 1.0, r, sigma, [0.25, 0.5, 0.75], [2.0, 2.0, 2.0], "put")
    assert abs(payoff.mean() - exact) < 4 * payoff.std() / np.sqrt(n)


def test_quadrature_converges():
    coarse = discrete_dividend_price(10.0, 10.0, 1.0, 0.05, 0.6, [0.9], [8.0], "put", n_nodes=16)
    fine = discrete_dividend_price(10.0, 10.0, 1.0, 0.05, 0.6, [0.9], [8.0], "put", n_nodes=48)
    assert coarse == pytest.approx(fine, abs=1e-8)


def test_escrowed_model_misprices_a_late_dividend():
    """Haug, Haug and Lewis's point: subtracting PV(D) from the spot is wrong when the dividend is near expiry."""
    exact = discrete_dividend_price(100.0, 100.0, 1.0, 0.05, 0.3, [0.99], [7.0])
    escrowed = black_scholes_price(100.0 - 7.0 * np.exp(-0.05 * 0.99), 100.0, 1.0, 0.05, 0.3)
    assert exact - escrowed > 0.3


def test_put_call_parity_with_cash_dividends():
    """c - p = S - PV(D) - K e^{-rT} while the stock can always pay the dividend."""
    c = discrete_dividend_price(100.0, 95.0, 1.0, 0.04, 0.2, [0.3, 0.8], [1.5, 1.5])
    p = discrete_dividend_price(100.0, 95.0, 1.0, 0.04, 0.2, [0.3, 0.8], [1.5, 1.5], "put")
    pv = 1.5 * (np.exp(-0.04 * 0.3) + np.exp(-0.04 * 0.8))
    # liquidation (S below 1.5) has probability ~1e-50 here, so parity holds to quadrature accuracy
    assert c - p == pytest.approx(100.0 - pv - 95.0 * np.exp(-0.04), abs=1e-9)


@pytest.mark.parametrize(
    "times,amounts,match",
    [
        ([0.5, 0.25], [1.0, 1.0], "increasing"),
        ([0.5], [1.0, 2.0], "same shape"),
        ([0.5], [-1.0], "dividends must be non-negative"),
        ([0.0], [1.0], "dividend_times must be positive"),
    ],
)
def test_invalid_dividend_schedules_raise(times, amounts, match):
    with pytest.raises(ValueError, match=match):
        discrete_dividend_price(100.0, 100.0, 1.0, 0.05, 0.3, times, amounts)
