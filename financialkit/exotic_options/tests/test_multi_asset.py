"""Tests for financialkit.exotic_options.systems.multi_asset."""

import numpy as np
import pytest

from financialkit.exotic_options import margrabe_price, quanto_price, stulz_price
from financialkit.integrators import correlated_increments
from financialkit.options import black_scholes_price

S1, S2, K, T, r, s1, s2, rho, q1, q2 = 100.0, 105.0, 98.0, 0.5, 0.05, 0.2, 0.3, 0.4, 0.01, 0.03


@pytest.fixture(scope="module")
def terminal():
    W = correlated_increments([[1.0, rho], [rho, 1.0]], T, 1, 200_000, rng=21)[:, 0, :]
    A = S1 * np.exp((r - q1 - 0.5 * s1**2) * T + s1 * W[:, 0])
    B = S2 * np.exp((r - q2 - 0.5 * s2**2) * T + s2 * W[:, 1])
    return A, B


def _covers(y, exact):
    return abs(y.mean() - exact) < 4 * y.std() / np.sqrt(y.size)


def test_margrabe_against_monte_carlo_and_black_scholes(terminal):
    A, B = terminal
    assert _covers(np.exp(-r * T) * np.maximum(A - B, 0.0), margrabe_price(S1, S2, T, s1, s2, rho, q1, q2))
    assert margrabe_price(42.0, 40.0, 0.5, 0.2, 0.0, 0.3, q2=0.10) == pytest.approx(black_scholes_price(42.0, 40.0, 0.5, 0.10, 0.2))


def test_margrabe_symmetry():
    """Exchanging B for A minus exchanging A for B is a forward on A - B."""
    diff = margrabe_price(S1, S2, T, s1, s2, rho, q1, q2) - margrabe_price(S2, S1, T, s2, s1, rho, q2, q1)
    assert diff == pytest.approx(S1 * np.exp(-q1 * T) - S2 * np.exp(-q2 * T), rel=1e-12)


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_stulz_against_monte_carlo(terminal, kind, option_type):
    A, B = terminal
    extreme = np.maximum(A, B) if kind == "max" else np.minimum(A, B)
    y = np.exp(-r * T) * np.maximum((1 if option_type == "call" else -1) * (extreme - K), 0.0)
    assert _covers(y, stulz_price(S1, S2, K, T, r, s1, s2, rho, q1, q2, option_type, kind))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_stulz_max_plus_min_is_two_vanillas(option_type):
    Ks = np.linspace(80.0, 130.0, 6)
    total = stulz_price(S1, S2, Ks, T, r, s1, s2, rho, q1, q2, option_type, "max") + stulz_price(S1, S2, Ks, T, r, s1, s2, rho, q1, q2, option_type, "min")
    vanilla = black_scholes_price(S1, Ks, T, r, s1, q1, option_type) + black_scholes_price(S2, Ks, T, r, s2, q2, option_type)
    np.testing.assert_allclose(total, vanilla, rtol=1e-10)
    with pytest.raises(ValueError):
        stulz_price(S1, S2, K, T, r, s1, s2, rho, kind="median")


def test_quanto_haug_example():
    """Haug (2007), Section 5.16: quanto call with fixed rate 1.5, 5.3280."""
    assert quanto_price(100.0, 105.0, 0.5, 0.08, 0.05, 0.2, 0.1, 0.3, q=0.04, fixed_rate=1.5) == pytest.approx(5.3280, abs=1e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_quanto_against_foreign_measure_monte_carlo(option_type):
    """Simulate under the foreign measure, where 1/X drifts at r_f - r_d and correlates -rho with S."""
    S, Kq, Tq, rd, rf, sS, sX, c, q, Xbar, X0 = 100.0, 105.0, 0.5, 0.08, 0.05, 0.2, 0.1, 0.3, 0.04, 1.5, 1.2
    W = correlated_increments([[1.0, -c], [-c, 1.0]], Tq, 1, 200_000, rng=22)[:, 0, :]
    ST = S * np.exp((rf - q - 0.5 * sS**2) * Tq + sS * W[:, 0])
    YT = np.exp((rf - rd - 0.5 * sX**2) * Tq + sX * W[:, 1]) / X0
    y = X0 * np.exp(-rf * Tq) * Xbar * np.maximum((1 if option_type == "call" else -1) * (ST - Kq), 0.0) * YT
    assert _covers(y, quanto_price(S, Kq, Tq, rd, rf, sS, sX, c, q, Xbar, option_type))
