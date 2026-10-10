"""Tests for the explicit, implicit and Crank-Nicolson schemes, grids and barrier options."""

import numpy as np
import pytest

from financialkit.options import black_scholes_greeks, black_scholes_price
from financialkit.pde_methods import (
    CrankNicolson,
    ExplicitScheme,
    FiniteDifferencePricer,
    ImplicitScheme,
    ThetaScheme,
    default_s_max,
    tavella_randall_grid,
)

ARGS = (100.0, 95.0, 1.0, 0.05, 0.25, 0.02)


def _images(S, K, H, T, r, sigma, q, option_type):
    """Method of images for a knock-out with no rebate (Hull, 10th ed., Section 26.9):
    V_out(S) = V(S) - (H/S)^(2 lambda - 2) V(H^2/S), lambda = (r - q + sigma^2/2)/sigma^2,
    valid for a down-and-out call with H <= K and an up-and-out put with H >= K."""
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    bs = lambda s: black_scholes_price(s, K, T, r, sigma, q, option_type)
    return bs(S) - (H / S) ** (2 * lam - 2) * bs(H**2 / S)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize(
    "scheme",
    [
        ExplicitScheme(200, 3000),
        ImplicitScheme(400, 1000),
        CrankNicolson(400, 200, rannacher_steps=2),
        ThetaScheme(400, 200, theta=0.6),
    ],
)
def test_european_prices_match_black_scholes(scheme, option_type):
    exact = black_scholes_price(*ARGS, option_type)
    res = scheme.price(*ARGS, option_type=option_type)
    # O(h^2 + dt) error with h = 1 on [0, 400]: below 2 cents for every scheme here
    assert res.price == pytest.approx(exact, abs=0.02)
    g = black_scholes_greeks(*ARGS, option_type)
    assert res.delta == pytest.approx(g.delta, abs=2e-3)
    assert res.gamma == pytest.approx(g.gamma, abs=2e-4)
    assert res.surface.shape == (scheme.n_time + 1, scheme.n_space + 1)
    np.testing.assert_allclose(res.surface[-1], np.maximum((1 if option_type == "call" else -1) * (res.S - 95.0), 0.0))


def test_schemes_are_finite_difference_pricers_with_method_names():
    assert isinstance(CrankNicolson(), FiniteDifferencePricer)
    assert [s.method for s in (ExplicitScheme(), ImplicitScheme(), CrankNicolson(), ThetaScheme(theta=0.3))] == [
        "explicit",
        "implicit",
        "crank_nicolson",
        "theta",
    ]


def test_put_call_parity_on_the_grid():
    S, K, T, r, sigma, q = ARGS
    cn = CrankNicolson(400, 200, rannacher_steps=2)
    c, p = cn.price(*ARGS, "call"), cn.price(*ARGS, "put")
    # the forward S e^{-q tau} - K e^{-r tau} solves the PDE exactly, but the discrete
    # time steps reproduce the exponentials only to O(dt^2) ~ 1e-6 here
    np.testing.assert_allclose(c.V - p.V, c.S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-5)


def test_crank_nicolson_is_second_order_in_space():
    exact = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)
    err = [abs(CrankNicolson(m, 400, rannacher_steps=2, s_max=400.0).price(100.0, 100.0, 1.0, 0.05, 0.2).price - exact) for m in (100, 200, 400)]
    assert np.log2(err[0] / err[1]) == pytest.approx(2.0, abs=0.15)
    assert np.log2(err[1] / err[2]) == pytest.approx(2.0, abs=0.25)


def test_implicit_scheme_is_first_order_in_time():
    exact = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)
    err = [ImplicitScheme(800, n, s_max=400.0).price(100.0, 100.0, 1.0, 0.05, 0.2).price - exact for n in (25, 50, 100)]
    assert np.log2(err[0] / err[1]) == pytest.approx(1.0, abs=0.1)
    assert np.log2(err[1] / err[2]) == pytest.approx(1.0, abs=0.1)


def test_rannacher_start_removes_crank_nicolson_gamma_oscillations():
    def gamma_error(rannacher):
        res = CrankNicolson(400, 10, rannacher_steps=rannacher, s_max=400.0).price(100.0, 100.0, 0.25, 0.05, 0.2)
        near = (res.S[1:-1] > 80) & (res.S[1:-1] < 120)
        gamma = np.diff(res.V, 2)[near]  # h = 1
        return np.abs(gamma - black_scholes_greeks(res.S[1:-1][near], 100.0, 0.25, 0.05, 0.2).gamma).max()

    # large steps leave CN's stiff modes undamped: the gamma error exceeds gamma itself (~0.04)
    assert gamma_error(0) > 0.02
    assert gamma_error(2) < 1e-3


def test_thomas_and_banded_solvers_give_the_same_prices():
    a = CrankNicolson(200, 50, solver="banded").price(*ARGS)
    b = CrankNicolson(200, 50, solver="thomas").price(*ARGS)
    np.testing.assert_allclose(a.V, b.V, atol=1e-10)


def test_tavella_randall_grid_beats_uniform_at_equal_size():
    exact = black_scholes_price(100.0, 100.0, 0.5, 0.05, 0.2)
    kw = dict(rannacher_steps=2, s_max=400.0)
    uni = abs(CrankNicolson(60, 400, **kw).price(100.0, 100.0, 0.5, 0.05, 0.2).price - exact)
    tr = abs(CrankNicolson(60, 400, grid="tavella_randall", concentration=0.15, **kw).price(100.0, 100.0, 0.5, 0.05, 0.2).price - exact)
    assert tr < 0.2 * uni


def test_tavella_randall_grid_shape():
    g = tavella_randall_grid(0.0, 400.0, 80, 100.0, 10.0)
    assert g[0] == 0.0 and g[-1] == 400.0 and np.all(np.diff(g) > 0)
    i = np.argmin(np.diff(g))
    assert g[i - 1] <= 100.0 <= g[i + 2]  # finest spacing at the strike
    with pytest.raises(ValueError):
        tavella_randall_grid(0.0, 400.0, 80, 500.0, 10.0)


def test_default_s_max():
    assert default_s_max(100.0, 80.0, 1.0, 0.2) == 400.0
    assert default_s_max(100.0, 100.0, 4.0, 0.5) == pytest.approx(100.0 * np.exp(5.0))


@pytest.mark.parametrize("grid", ["uniform", "tavella_randall"])
def test_down_and_out_call_matches_method_of_images(grid):
    S, K, H, T, r, sigma, q = 100.0, 100.0, 90.0, 1.0, 0.05, 0.25, 0.01
    res = CrankNicolson(400, 200, rannacher_steps=2, grid=grid).price_barrier(S, K, H, T, r, sigma, q, "call", "down-and-out")
    assert res.S[0] == H and res.V[0] == 0.0
    assert res.price == pytest.approx(_images(S, K, H, T, r, sigma, q, "call"), abs=0.01)


def test_up_and_out_put_matches_method_of_images():
    S, K, H, T, r, sigma = 100.0, 100.0, 115.0, 0.5, 0.05, 0.2
    res = CrankNicolson(400, 200, rannacher_steps=2).price_barrier(S, K, H, T, r, sigma, 0.0, "put", "up-and-out")
    assert res.S[-1] == H
    assert res.price == pytest.approx(_images(S, K, H, T, r, sigma, 0.0, "put"), abs=0.01)


def test_input_validation():
    with pytest.raises(ValueError):
        ThetaScheme(theta=1.5)
    with pytest.raises(ValueError):
        ThetaScheme(grid="chebyshev")
    with pytest.raises(ValueError):
        ThetaScheme(american_method="lsm")
    with pytest.raises(ValueError):
        CrankNicolson(n_space=1)
    with pytest.raises(ValueError):
        CrankNicolson().price(-1.0, 100.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError):
        CrankNicolson(s_max=50.0).price(100.0, 100.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError):
        CrankNicolson().price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="straddle")
    with pytest.raises(ValueError):
        CrankNicolson().price_barrier(80.0, 100.0, 90.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError):
        CrankNicolson().price_barrier(100.0, 100.0, 90.0, 1.0, 0.05, 0.2, barrier_type="down-and-in")
