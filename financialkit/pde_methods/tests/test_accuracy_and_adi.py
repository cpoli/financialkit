"""Tests for Richardson extrapolation and the Peaceman-Rachford ADI scheme."""

import numpy as np
import pytest

from financialkit.options import black_scholes_price
from financialkit.pde_methods import CrankNicolson, PeacemanRachfordADI, richardson_extrapolation


def test_richardson_is_exact_for_a_pure_power_error():
    h = np.array([0.4, 0.2, 0.1])
    res = richardson_extrapolation(2.0 + 3.0 * h**2)
    assert res.extrapolated == pytest.approx(2.0, abs=1e-12)
    assert res.observed_order == pytest.approx(2.0, abs=1e-12) and res.order == res.observed_order
    first = richardson_extrapolation(2.0 + 3.0 * h, order=1.0)
    assert first.extrapolated == pytest.approx(2.0, abs=1e-12)
    assert np.isnan(richardson_extrapolation([1.0, 1.1]).observed_order)
    with pytest.raises(ValueError):
        richardson_extrapolation([1.0])
    with pytest.raises(ValueError):
        richardson_extrapolation([1.0, 2.0], ratio=1.0)


def test_richardson_improves_crank_nicolson_by_an_order_of_magnitude():
    exact = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)
    prices = [CrankNicolson(m, m, rannacher_steps=2, s_max=400.0).price(100.0, 100.0, 1.0, 0.05, 0.2).price for m in (40, 80, 160)]
    res = richardson_extrapolation(prices)
    assert res.observed_order == pytest.approx(2.0, abs=0.2)
    assert abs(res.extrapolated - exact) < 0.1 * abs(prices[-1] - exact)


def _margrabe(S1, S2, T, sigma1, sigma2, rho, q1=0.0, q2=0.0):
    """Margrabe (1978): the exchange option is Black-Scholes with strike S2, rate q2 and the spread volatility."""
    return black_scholes_price(S1, S2, T, q2, np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2), q1)


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.5])
def test_adi_exchange_option_matches_margrabe(rho):
    adi = PeacemanRachfordADI(120, 120, 60)
    res = adi.price(100.0, 95.0, 1.0, 0.05, 0.2, 0.3, rho, lambda a, b: np.maximum(a - b, 0.0), q1=0.02)
    # O(dx^2 + dt^2) error, plus the boundary approximation five standard deviations out
    assert res.price == pytest.approx(_margrabe(100.0, 95.0, 1.0, 0.2, 0.3, rho, q1=0.02), abs=0.01)
    assert res.V.shape == (121, 121) and res.n_time == 60


def test_adi_separable_payoff_with_zero_correlation():
    # with rho = 0 a call on S1 alone ignores S2 and is the one-asset Black-Scholes price
    res = PeacemanRachfordADI(100, 40, 50).price(100.0, 50.0, 1.0, 0.05, 0.25, 0.3, 0.0, lambda a, b: np.maximum(a - 100.0, 0.0))
    # dx = 0.025 in log-spot: O(dx^2) error of about a cent
    assert res.price == pytest.approx(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.25), abs=0.02)


def test_adi_validation():
    with pytest.raises(ValueError):
        PeacemanRachfordADI(n1=1)
    with pytest.raises(ValueError):
        PeacemanRachfordADI().price(100.0, 100.0, 1.0, 0.05, 0.2, 0.2, 1.5, lambda a, b: a)
    assert PeacemanRachfordADI(51, 51).n1 == 52


def test_adi_rejects_non_positive_inputs():
    with pytest.raises(ValueError, match="must all be positive"):
        PeacemanRachfordADI(n1=10, n2=10, n_time=5).price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 0.5, lambda a, b: np.maximum(a - b, 0.0))
