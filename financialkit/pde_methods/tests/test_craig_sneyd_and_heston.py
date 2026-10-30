"""Tests for the ADI splitting schemes, the Craig-Sneyd two-asset pricer and the Heston PDE."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.integrate import quad

from financialkit.options import black_scholes_price
from financialkit.pde_methods import (
    CraigSneydADI,
    HestonADI,
    adi_march,
    default_adi_theta,
    difference_matrices,
    heston_spot_grid,
    heston_variance_grid,
)


def _scalar(a0, a1, a2):
    return [sp.csr_matrix([[a]]) for a in (a0, a1, a2)]


def _no_bc(tau):
    return [np.zeros(1)] * 3


@pytest.mark.parametrize(("scheme", "order"), [("douglas", 1), ("craig_sneyd", 2), ("modified_craig_sneyd", 2), ("hundsdorfer_verwer", 2)])
def test_adi_march_order_on_a_scalar_ode_with_a_mixed_term(scheme, order):
    # U' = (a0 + a1 + a2) U: the explicit a0 makes Douglas first order (Craig-Sneyd 1988, in 't Hout-Foulon 2010)
    A = _scalar(-0.6, -0.5, -0.4)
    errs = [abs(adi_march(A, _no_bc, [1.0], 1.0, n, scheme)[0] - np.exp(-1.5)) for n in (40, 80)]
    assert np.log2(errs[0] / errs[1]) == pytest.approx(order, abs=0.15)


def test_adi_march_boundary_terms_and_validation():
    # U' = -U + 1 has U(1) = 1 - (1 - U0) e^{-1}; split the source between the directions
    A = _scalar(0.0, -0.5, -0.5)
    U = adi_march(A, lambda tau: [np.zeros(1), np.full(1, 0.5), np.full(1, 0.5)], [0.0], 1.0, 200)
    assert U[0] == pytest.approx(1.0 - np.exp(-1.0), abs=1e-5)
    with pytest.raises(ValueError, match="scheme"):
        adi_march(A, _no_bc, [1.0], 1.0, 10, scheme="strang")
    with pytest.raises(ValueError, match="theta"):
        adi_march(A, _no_bc, [1.0], 1.0, 10, theta=0.0)
    with pytest.raises(ValueError, match="n_time"):
        adi_march(A, _no_bc, [1.0], 1.0, 0)
    with pytest.raises(ValueError):
        default_adi_theta("strang")
    assert default_adi_theta("hundsdorfer_verwer") == pytest.approx(0.5 + np.sqrt(3) / 6)


def test_difference_matrices_are_exact_on_quadratics():
    x = np.cumsum(np.r_[0.0, np.random.default_rng(0).uniform(0.5, 1.5, 9)])
    D1, D2 = difference_matrices(x)
    assert np.allclose((D1 @ x**2)[1:-1], 2 * x[1:-1]) and np.allclose((D2 @ x**2)[1:-1], 2.0)
    assert D1[0].nnz == 0 and D2[-1].nnz == 0
    with pytest.raises(ValueError):
        difference_matrices([0.0, 1.0])


def _margrabe(S1, S2, T, sigma1, sigma2, rho):
    """Margrabe (1978): Black-Scholes with strike S2, zero rate and the spread volatility."""
    return black_scholes_price(S1, S2, T, 0.0, np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2))


def _exchange(a, b):
    return np.maximum(a - b, 0.0)


@pytest.mark.parametrize("rho", [-0.9, 0.0, 0.5])
def test_craig_sneyd_exchange_option_matches_margrabe(rho):
    res = CraigSneydADI(120, 120, 40).price(100.0, 95.0, 1.0, 0.05, 0.2, 0.3, rho, _exchange)
    # O(dx^2 + dt^2) error; at rho -> 1 the spread volatility shrinks and the axis-aligned grid needs refining
    assert res.price == pytest.approx(_margrabe(100.0, 95.0, 1.0, 0.2, 0.3, rho), abs=0.02)
    assert res.V.shape == (121, 121) and res.method == "craig_sneyd" and res.n_time == 40


def test_craig_sneyd_is_second_order_where_douglas_is_first():
    """Temporal error against a 640-step solve on the same grid (Craig and Sneyd 1988)."""

    def price(n, scheme):
        return CraigSneydADI(60, 60, n, scheme=scheme).price(100.0, 95.0, 1.0, 0.05, 0.2, 0.3, 0.5, _exchange).price

    ref = price(640, "craig_sneyd")
    for scheme, order in (("douglas", 1.0), ("craig_sneyd", 2.0)):
        e1, e2 = abs(price(20, scheme) - ref), abs(price(40, scheme) - ref)
        assert np.log2(e1 / e2) == pytest.approx(order, abs=0.3)


def test_craig_sneyd_validation():
    with pytest.raises(ValueError):
        CraigSneydADI(n1=1)
    with pytest.raises(ValueError):
        CraigSneydADI(scheme="strang")
    cs = CraigSneydADI(10, 10, 5)
    assert cs.adi_theta == 0.5 and CraigSneydADI(11, 11).n1 == 12
    with pytest.raises(ValueError, match="rho"):
        cs.price(100.0, 100.0, 1.0, 0.05, 0.2, 0.2, 1.5, _exchange)
    with pytest.raises(ValueError, match="positive"):
        cs.price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 0.5, _exchange)


def _heston_call(S, K, T, r, q, v0, kappa, theta, xi, rho):
    """Heston (1993) two-probability formula with the "little trap" characteristic function of Albrecher et al. (2007)."""

    def cf(u):
        beta = kappa - rho * xi * 1j * u
        d = np.sqrt(beta**2 + xi**2 * (1j * u + u**2))
        g = (beta - d) / (beta + d)
        e = np.exp(-d * T)
        C = kappa * theta / xi**2 * ((beta - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
        return np.exp(C + (beta - d) / xi**2 * (1.0 - e) / (1.0 - g * e) * v0)

    F = S * np.exp((r - q) * T)
    k = np.log(F / K)

    def prob(shift):
        return 0.5 + quad(lambda u: np.real(np.exp(1j * u * k) * cf(u + shift) / (1j * u)), 0.0, np.inf, limit=2000)[0] / np.pi

    return np.exp(-r * T) * (F * prob(-1j) - K * prob(0.0))


# cases A-D of in 't Hout and Foulon (2010): kappa, theta, xi, rho, r, q, T
HESTON_CASES = {
    "A": (1.5, 0.04, 0.3, -0.9, 0.025, 0.0, 1.0),
    "B": (3.0, 0.12, 0.04, 0.6, 0.01, 0.04, 1.0),
    "C": (0.6067, 0.0707, 0.2928, -0.7571, 0.03, 0.0, 3.0),
    "D": (2.5, 0.06, 0.5, -0.1, 0.0507, 0.0469, 0.25),
}


@pytest.mark.parametrize("case", sorted(HESTON_CASES))
def test_heston_adi_matches_the_heston_formula(case):
    kappa, theta, xi, rho, r, q, T = HESTON_CASES[case]
    exact = _heston_call(100.0, 100.0, T, r, q, theta, kappa, theta, xi, rho)
    coarse = HestonADI(50, 25, 20).price(100.0, 100.0, T, r, theta, kappa, theta, xi, rho, q)
    fine = HestonADI(100, 50, 50).price(100.0, 100.0, T, r, theta, kappa, theta, xi, rho, q)
    # second order in space: about 1.5 cents on the 50 x 25 grid, a few tenths of a cent on 100 x 50
    assert abs(coarse.price - exact) < 0.02
    assert abs(fine.price - exact) < 0.5 * abs(coarse.price - exact)
    assert 0.0 < fine.delta < 1.0


def test_heston_adi_when_the_feller_condition_fails():
    """Andersen (2008), Case I: 2 kappa theta = 0.04 < xi^2 = 1, and the ten-year at-the-money call is worth 13.0847."""
    res = HestonADI(200, 100, 100).price(100.0, 100.0, 10.0, 0.0, 0.04, 0.5, 0.04, 1.0, -0.9)
    # about 4e-3 on this grid, mostly from the spot direction (8e-3 with m1 = 100)
    assert res.price == pytest.approx(13.0847, abs=0.01)


def test_heston_adi_put_call_parity_and_schemes():
    args = (100.0, 100.0, 1.0, 0.025, 0.04, 1.5, 0.04, 0.3, -0.9)
    call = HestonADI(40, 20, 10).price(*args)
    put = HestonADI(40, 20, 10).price(*args, option_type="put")
    assert put.price == pytest.approx(call.price - 100.0 + 100.0 * np.exp(-0.025), abs=1e-10)
    assert put.delta == pytest.approx(call.delta - 1.0, abs=1e-8)
    prices = [HestonADI(40, 20, 40, scheme=s).price(*args).price for s in ("douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer")]
    assert np.ptp(prices) < 0.02
    assert call.V.shape == (41, 21) and call.method == "modified_craig_sneyd"
    assert np.all(call.V[0] == 0.0) and np.allclose(call.V[:, -1], call.S)


def test_heston_grids_and_validation():
    s = heston_spot_grid(40, 100.0, 800.0, T=1.0)
    ds = np.diff(s)
    assert s[0] == 0.0 and s[-1] == 800.0 and np.all(ds > 0)
    uniform = (s[1:] <= 100.0 + 1e-9) & (s[:-1] >= np.exp(-0.25) * 100.0 - 1e-9)
    assert uniform.any() and np.allclose(ds[uniform], ds[uniform][0])
    v = heston_variance_grid(20, 5.0)
    assert v[0] == 0.0 and v[-1] == 5.0 and np.all(np.diff(np.diff(v)) > 0)
    for bad in (
        lambda: heston_spot_grid(2, 100.0, 800.0),
        lambda: heston_spot_grid(10, 100.0, 50.0),
        lambda: heston_variance_grid(2),
        lambda: heston_variance_grid(10, 0.0),
    ):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        HestonADI(n_time=0)
    with pytest.raises(ValueError):
        HestonADI(s_max_factor=1.0)
    h = HestonADI(20, 10, 5)
    with pytest.raises(ValueError, match="rho"):
        h.price(100.0, 100.0, 1.0, 0.0, 0.04, 1.5, 0.04, 0.3, -1.5)
    with pytest.raises(ValueError, match="kappa"):
        h.price(100.0, 100.0, 1.0, 0.0, 0.04, 0.0, 0.04, 0.3, -0.5)
    with pytest.raises(ValueError, match="inside the grid"):
        h.price(100.0, 100.0, 1.0, 0.0, 6.0, 1.5, 0.04, 0.3, -0.5)
