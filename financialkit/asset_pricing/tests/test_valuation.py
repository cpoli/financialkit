"""Tests for the dividend discount model, Gordon growth, Modigliani-Miller and Shiller's bound."""

import numpy as np
import pytest

from financialkit.asset_pricing import (
    ar1_dividend_price,
    campbell_shiller_decomposition,
    campbell_shiller_rho,
    dividend_discount_value,
    ex_post_rational_price,
    gordon_growth_price,
    implied_cost_of_equity,
    levered_cost_of_equity,
    levered_firm_value,
    log_linear_return,
    sustainable_growth,
    variance_bound,
    wacc,
)
from financialkit.constants import DEFAULT_RTOL


def test_ddm_with_constant_growth_is_gordon():
    D1, r, g = 2.0, 0.09, 0.04
    divs = D1 * (1 + g) ** np.arange(5)
    assert dividend_discount_value(divs, r, terminal_growth=g) == pytest.approx(gordon_growth_price(D1, r, g), rel=DEFAULT_RTOL)


def test_ddm_finite_horizon_converges_to_gordon():
    D1, r, g = 1.0, 0.08, 0.03
    long = dividend_discount_value(D1 * (1 + g) ** np.arange(2000), r)
    assert long == pytest.approx(gordon_growth_price(D1, r, g), rel=1e-9)


def test_gordon_round_trip_and_broadcast():
    P = gordon_growth_price(2.0, 0.09, np.array([0.0, 0.04]))
    np.testing.assert_allclose(P, [2 / 0.09, 40.0])
    assert implied_cost_of_equity(40.0, 2.0, 0.04) == pytest.approx(0.09)
    assert sustainable_growth(0.15, 0.6) == pytest.approx(0.06)
    with pytest.raises(ValueError):
        gordon_growth_price(1.0, 0.05, 0.06)


def test_modigliani_miller_propositions():
    rA, rD = 0.10, 0.05
    de = np.linspace(0.0, 3.0, 7)
    rE = levered_cost_of_equity(rA, rD, de)
    # Proposition II without taxes makes the WACC equal the asset return at every leverage.
    np.testing.assert_allclose(wacc(rE, rD, de / (1 + de)), rA)
    assert levered_firm_value(100.0, 40.0) == 100.0
    assert levered_firm_value(100.0, 40.0, 0.25) == pytest.approx(110.0)


def test_modigliani_miller_with_taxes_consistent_values():
    """With perpetual debt, V_L = V_U + tD, and E r_E = (V_U r_A - ...) per Modigliani-Miller (1963)."""
    t, rA, rD, VU, D = 0.3, 0.1, 0.05, 100.0, 40.0
    VL = levered_firm_value(VU, D, t)
    E = VL - D
    rE = levered_cost_of_equity(rA, rD, D / E, t)
    # Equity holders receive after-tax operating income minus after-tax interest.
    ebit_after_tax = rA * VU
    assert rE * E == pytest.approx(ebit_after_tax - (1 - t) * rD * D)


def test_ex_post_rational_price_recursion():
    D = np.array([0.0, 1.0, 2.0, 3.0])
    p = ex_post_rational_price(D, 0.1, 10.0)
    assert p[2] == pytest.approx((10 + 3) / 1.1)
    assert p[0] == pytest.approx(1 / 1.1 + 2 / 1.1**2 + 13 / 1.1**3)


def test_ar1_price_satisfies_the_pricing_recursion():
    r, rho, Dbar = 0.05, 0.8, 1.0
    D = np.array([0.5, 1.0, 1.7])
    lhs = ar1_dividend_price(D, r, rho, Dbar)
    ED_next = Dbar + rho * (D - Dbar)
    rhs = (ar1_dividend_price(ED_next, r, rho, Dbar) + ED_next) / (1 + r)
    np.testing.assert_allclose(lhs, rhs, rtol=DEFAULT_RTOL)


def test_rational_price_respects_the_variance_bound():
    rng = np.random.default_rng(0)
    r, rho, Dbar, n = 0.05, 0.9, 1.0, 20000
    D = np.empty(n)
    D[0] = Dbar
    for t in range(1, n):
        D[t] = Dbar + rho * (D[t - 1] - Dbar) + 0.05 * rng.standard_normal()
    good = variance_bound(ar1_dividend_price(D, r, rho, Dbar), D, r, terminal_price=Dbar / r)
    assert not good.violated
    bad = variance_bound(D / r, D, r, terminal_price=Dbar / r)
    assert bad.violated and bad.ratio > 1.5


def test_campbell_shiller_rho_and_linear_return():
    rho = campbell_shiller_rho(np.log(0.04))
    assert rho == pytest.approx(1 / 1.04)
    dp = np.log(0.04) + np.array([-0.2, 0.0, 0.2])
    exact = 0.02 + np.log1p(np.exp(-dp)) + np.log(0.04)
    approx = log_linear_return(np.log(0.04), dp, 0.02, rho)
    # Exact at the mean; the error is second order in the deviation.
    assert approx[1] == pytest.approx(exact[1])
    err = np.abs(approx - exact)
    assert err[1] < 1e-12 and 0 < err[0] < 0.01 and 0 < err[2] < 0.01


def test_campbell_shiller_decomposition_identity():
    """With unpredictable dividends, long-run return coefficient 1 and dividend coefficient 0 (Cochrane 2008)."""
    rng = np.random.default_rng(0)
    T, phi, m = 20_000, 0.9, np.log(0.04)
    dp = np.empty(T)
    dp[0] = m
    for t in range(1, T):
        dp[t] = m + phi * (dp[t - 1] - m) + 0.05 * rng.standard_normal()
    dd = 0.02 + 0.1 * rng.standard_normal(T)
    r = np.r_[np.nan, dd[1:] + np.log1p(np.exp(-dp[1:])) + dp[:-1]]
    pv = campbell_shiller_decomposition(r, dd, dp)
    assert abs(pv.identity_error) < 0.01
    assert pv.long_run_return == pytest.approx(1.0, abs=0.1)
    assert abs(pv.long_run_dividend) < 0.1
