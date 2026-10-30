"""Tests for financialkit.monte_carlo.systems.adjoint (Giles-Glasserman adjoint Greeks)."""

import numpy as np
import pytest

from financialkit.monte_carlo import LMMGreeks, lmm_swaption_greeks

N, M, DELTA = 10, 4, 0.25
L0 = 0.05 + 0.002 * np.arange(N)
SIGMA = 0.22 - 0.01 * np.arange(N)
BOOK = [(4, 0.05), (6, 0.055), (2, 0.045)]
Z = np.random.default_rng(7).standard_normal((1500, M))


def _price(L0=L0, sigma=SIGMA):
    return lmm_swaption_greeks(L0, sigma, DELTA, M, BOOK, z=Z).price


def test_adjoint_equals_forward_mode():
    adj = lmm_swaption_greeks(L0, SIGMA, DELTA, M, BOOK, z=Z)
    fwd = lmm_swaption_greeks(L0, SIGMA, DELTA, M, BOOK, z=Z, method="forward")
    assert isinstance(adj, LMMGreeks) and fwd.method == "forward"
    assert fwd.price == adj.price
    # Same chain rule in a different order: equal up to rounding.
    np.testing.assert_allclose(adj.delta, fwd.delta, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(adj.vega, fwd.vega, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("wrt", ["L0", "sigma"])
def test_matches_central_differences_on_common_numbers(wrt):
    g = lmm_swaption_greeks(L0, SIGMA, DELTA, M, BOOK, z=Z)
    h = 1e-6
    base = L0 if wrt == "L0" else SIGMA
    fd = np.array([(_price(**{wrt: base + h * e}) - _price(**{wrt: base - h * e})) / (2 * h) for e in np.eye(N)])
    # Central-difference error O(h^2) on a smooth sample mean, well below 1e-7.
    np.testing.assert_allclose(g.delta if wrt == "L0" else g.vega, fd, atol=1e-7)


def test_fixed_rates_only_discount():
    """Rates fixed before expiry enter through the numeraire alone, so their deltas are negative."""
    g = lmm_swaption_greeks(L0, SIGMA, DELTA, M, BOOK, n_paths=4000, rng=3)
    assert np.all(g.delta[:M] < 0) and np.all(g.delta[M : M + 2] > 0)
    assert g.vega[0] == 0.0  # L_0 fixes at time zero
    assert np.all(g.delta_se > 0) and g.price_se > 0 and g.n_paths == 4000


def test_deterministic_limit_is_intrinsic():
    """With zero volatility the forwards are frozen and the book pays its intrinsic value."""
    g = lmm_swaption_greeks(L0, 0.0, DELTA, M, BOOK, n_paths=10, rng=0)
    B = np.prod(1 + DELTA * L0[:M])
    P = np.cumprod(1 / (1 + DELTA * L0[M:]))
    value = sum(max(0.0, np.sum(DELTA * (L0[M : M + k] - K) * P[:k])) for k, K in BOOK) / B
    assert g.price == pytest.approx(value, rel=1e-12)
    assert g.price_se == 0.0


def test_chunking_does_not_change_the_result(monkeypatch):
    import financialkit.monte_carlo.systems.adjoint as adjoint

    full = lmm_swaption_greeks(L0, SIGMA, DELTA, M, BOOK, z=Z)
    monkeypatch.setattr(adjoint, "_CHUNK_FLOATS", 7 * N * (M + 1))
    small = lmm_swaption_greeks(L0, SIGMA, DELTA, M, BOOK, z=Z)
    np.testing.assert_allclose(small.delta, full.delta, rtol=1e-12)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"expiry": 0}, "expiry"),
        ({"expiry": N}, "expiry"),
        ({"swaptions": []}, "empty"),
        ({"swaptions": [(N, 0.05)]}, "swap length"),
        ({"z": np.zeros((5, M + 1))}, "z must have shape"),
        ({"method": "reverse"}, "method"),
        ({"delta": 0.0}, "delta must be positive"),
        ({"sigma": -0.1}, "sigma must be non-negative"),
        ({"L0": -L0}, "L0 must be positive"),
    ],
)
def test_validation(kwargs, match):
    args = {"L0": L0, "sigma": SIGMA, "delta": DELTA, "expiry": M, "swaptions": BOOK, "n_paths": 5} | kwargs
    with pytest.raises(ValueError, match=match):
        lmm_swaption_greeks(**args)
