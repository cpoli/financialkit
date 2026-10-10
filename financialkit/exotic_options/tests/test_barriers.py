"""Tests for financialkit.exotic_options.systems.barriers."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.exotic_options import (
    BGK_BETA,
    barrier_option_mc,
    barrier_price,
    discrete_barrier_price,
    double_barrier_price,
    merton_down_and_out_call,
)
from financialkit.options import black_scholes_price

# Haug, *The Complete Guide to Option Pricing Formulas*, 2nd ed., Table 4-13:
# S = 100, T = 0.5, r = 8%, b = 4% (q = 4%), rebate 3, sigma = 25%,
# H = 95 for down barriers and 105 for up barriers, K = 90, 100, 110.
HAUG_4_13 = {
    ("call", "down-and-out", 95.0): (9.0246, 6.7924, 4.8759),
    ("call", "up-and-out", 105.0): (2.6789, 2.3580, 2.3453),
    ("call", "down-and-in", 95.0): (7.7627, 4.0109, 2.0576),
    ("call", "up-and-in", 105.0): (14.1112, 8.4482, 4.5910),
    ("put", "down-and-out", 95.0): (2.2798, 2.2947, 2.6252),
    ("put", "up-and-out", 105.0): (3.7760, 5.4932, 7.5187),
    ("put", "down-and-in", 95.0): (2.9586, 6.5677, 11.9752),
    ("put", "up-and-in", 105.0): (1.4653, 3.3721, 7.0846),
}


@pytest.mark.parametrize(("option_type", "barrier_type", "H"), list(HAUG_4_13))
def test_reiner_rubinstein_matches_haug_table_4_13(option_type, barrier_type, H):
    """Haug (2007), Table 4-13, sigma = 25% column (printed to 4 decimals)."""
    got = barrier_price(100.0, np.array([90.0, 100.0, 110.0]), H, 0.5, 0.08, 0.25, q=0.04, option_type=option_type, barrier_type=barrier_type, rebate=3.0)
    np.testing.assert_allclose(got, HAUG_4_13[(option_type, barrier_type, H)], atol=1e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("direction", ["down", "up"])
def test_in_out_parity(option_type, direction):
    K = np.linspace(70.0, 130.0, 13)
    H = 85.0 if direction == "down" else 120.0
    args = (100.0, K, H, 0.75, 0.05, 0.3, 0.02, option_type)
    total = barrier_price(*args, f"{direction}-and-out") + barrier_price(*args, f"{direction}-and-in")
    np.testing.assert_allclose(total, black_scholes_price(100.0, K, 0.75, 0.05, 0.3, 0.02, option_type), rtol=DEFAULT_RTOL)


def test_merton_down_and_out_equals_reiner_rubinstein():
    S = np.linspace(81.0, 140.0, 12)
    for K in (80.0, 100.0, 120.0):
        np.testing.assert_allclose(
            merton_down_and_out_call(S, K, 80.0, 1.0, 0.05, 0.25, 0.01), barrier_price(S, K, 80.0, 1.0, 0.05, 0.25, 0.01), rtol=DEFAULT_RTOL
        )
    assert merton_down_and_out_call(79.0, 100.0, 80.0, 1.0, 0.05, 0.25) == 0.0
    with pytest.raises(ValueError):
        merton_down_and_out_call(100.0, 90.0, 95.0, 1.0, 0.05, 0.25)


def test_knocked_spot():
    assert barrier_price(90.0, 100.0, 95.0, 1.0, 0.05, 0.2, rebate=2.0) == 2.0
    assert barrier_price(90.0, 100.0, 95.0, 1.0, 0.05, 0.2, barrier_type="down-and-in") == pytest.approx(black_scholes_price(90.0, 100.0, 1.0, 0.05, 0.2))
    with pytest.raises(ValueError):
        barrier_price(100.0, 100.0, 95.0, 1.0, 0.05, 0.2, barrier_type="sideways")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_double_barrier_reduces_to_single_barriers(option_type):
    """A remote upper (lower) barrier leaves the down-and-out (up-and-out) option."""
    K = np.array([85.0, 100.0, 115.0])
    args = (0.5, 0.06, 0.25, 0.02, option_type)
    lone_down = double_barrier_price(100.0, K, 80.0, 1e5, *args)
    lone_up = double_barrier_price(100.0, K, 1e-5, 125.0, *args)
    np.testing.assert_allclose(lone_down, barrier_price(100.0, K, 80.0, *args[:4], option_type, "down-and-out"), atol=1e-10)
    np.testing.assert_allclose(lone_up, barrier_price(100.0, K, 125.0, *args[:4], option_type, "up-and-out"), atol=1e-10)


def test_double_barrier_in_out_and_series_convergence():
    args = (100.0, 100.0, 80.0, 120.0, 1.0, 0.05, 0.3)
    out = double_barrier_price(*args)
    assert out == pytest.approx(double_barrier_price(*args, n_terms=20), rel=1e-12)
    assert out + double_barrier_price(*args, knock="in") == pytest.approx(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.3))
    assert double_barrier_price(130.0, 100.0, 80.0, 120.0, 1.0, 0.05, 0.3) == 0.0


def test_double_barrier_monte_carlo():
    """Continuous double barrier against bridge-corrected Monte Carlo of both barriers."""
    from financialkit.exotic_options import barrier_crossing_probability
    from financialkit.integrators import gbm_paths

    S, K, L, U, T, r, sigma = 100.0, 100.0, 85.0, 125.0, 0.5, 0.05, 0.2
    n = 50
    _, paths = gbm_paths(S, r, sigma, T, n, 100_000, rng=7)
    x = np.log(paths)
    p_lo = barrier_crossing_probability(x[:, :-1], x[:, 1:], np.log(L), sigma, T / n)
    p_hi = barrier_crossing_probability(x[:, :-1], x[:, 1:], np.log(U), sigma, T / n)
    # Crossing both barriers within one step is negligible at this step size.
    y = np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0) * np.prod(1.0 - np.minimum(p_lo + p_hi, 1.0), axis=1)
    exact = double_barrier_price(S, K, L, U, T, r, sigma)
    assert abs(y.mean() - exact) < 4 * y.std() / np.sqrt(y.size)


@pytest.mark.parametrize(("option_type", "barrier_type", "H"), [("call", "down-and-out", 90.0), ("put", "up-and-in", 110.0), ("call", "up-and-out", 130.0)])
def test_bridge_monte_carlo_prices_continuous_barrier(option_type, barrier_type, H):
    exact = barrier_price(100.0, 100.0, H, 1.0, 0.05, 0.25, 0.01, option_type, barrier_type)
    res = barrier_option_mc(100.0, 100.0, H, 1.0, 0.05, 0.25, 0.01, option_type, barrier_type, n_steps=25, n_paths=100_000, rng=1)
    assert abs(res.estimate - exact) < 4 * res.std_error


def test_bgk_constant():
    from scipy.special import zeta

    assert BGK_BETA == pytest.approx(-zeta(0.5) / np.sqrt(2 * np.pi), rel=1e-12)


@pytest.mark.parametrize(("barrier_type", "H"), [("down-and-out", 90.0), ("up-and-out", 120.0)])
def test_bgk_correction_prices_discrete_monitoring(barrier_type, H):
    """Broadie-Glasserman-Kou (1997): the shifted barrier matches discrete monitoring;
    the continuous price is off by many standard errors."""
    m = 50
    res = barrier_option_mc(100.0, 100.0, H, 1.0, 0.05, 0.3, n_steps=m, n_paths=200_000, continuous=False, rng=2, barrier_type=barrier_type)
    corrected = discrete_barrier_price(100.0, 100.0, H, 1.0, 0.05, 0.3, barrier_type=barrier_type, n_monitor=m)
    continuous = barrier_price(100.0, 100.0, H, 1.0, 0.05, 0.3, barrier_type=barrier_type)
    # o(1/sqrt(m)) correction error: allow one cent beyond the sampling noise.
    assert abs(res.estimate - corrected) < 4 * res.std_error + 0.01
    assert abs(res.estimate - continuous) > 10 * res.std_error
