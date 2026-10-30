"""Published reference values for the options domain (ROADMAP B7).

Every number below is printed in Haug, *The Complete Guide to Option
Pricing Formulas*, 2nd ed. (McGraw-Hill, 2007), in the generalized
Black-Scholes-Merton notation with cost of carry ``b = r - q``. The
tolerance is half a unit in the last printed digit, so a perturbed model
fails. Hull's examples for the same pricers are in ``test_closed_form.py``,
``test_greeks.py``, ``test_implied_vol.py`` and ``test_trees.py``.
"""

import pytest

from financialkit.options import black76_price, black_scholes_greeks, black_scholes_price

HALF_ULP_4DP = 5e-5


def test_black_scholes_haug_section_1_1_1():
    """Haug, 2nd ed., Section 1.1.1: S = 60, X = 65, T = 0.25, r = 8%, sigma = 30% gives c = 2.1334."""
    assert black_scholes_price(60.0, 65.0, 0.25, 0.08, 0.30) == pytest.approx(2.1334, abs=HALF_ULP_4DP)


def test_merton_index_put_haug_section_1_1_2():
    """Haug, 2nd ed., Section 1.1.2: S = 100, X = 95, T = 0.5, r = 10%, q = 5%, sigma = 20% gives p = 2.4648."""
    put = black_scholes_price(100.0, 95.0, 0.5, 0.10, 0.20, q=0.05, option_type="put")
    assert put == pytest.approx(2.4648, abs=HALF_ULP_4DP)


def test_black76_haug_section_1_1_3():
    """Haug, 2nd ed., Section 1.1.3: F = X = 19, T = 0.75, r = 10%, sigma = 28% gives c = p = 1.7011."""
    for option_type in ("call", "put"):
        assert black76_price(19.0, 19.0, 0.75, 0.10, 0.28, option_type) == pytest.approx(1.7011, abs=HALF_ULP_4DP)


def test_delta_haug_section_2_1_1():
    """Haug, 2nd ed., Section 2.1.1: S = 105, X = 100, T = 0.5, r = 10%, b = 0, sigma = 36%
    gives call delta 0.5946 and put delta -0.3566."""
    call = black_scholes_greeks(105.0, 100.0, 0.5, 0.10, 0.36, q=0.10)
    put = black_scholes_greeks(105.0, 100.0, 0.5, 0.10, 0.36, q=0.10, option_type="put")
    assert call.delta == pytest.approx(0.5946, abs=HALF_ULP_4DP)
    assert put.delta == pytest.approx(-0.3566, abs=HALF_ULP_4DP)


def test_gamma_haug_section_2_2_1():
    """Haug, 2nd ed., Section 2.2.1: S = 55, X = 60, T = 0.75, r = b = 10%, sigma = 30% gives gamma 0.0278."""
    assert black_scholes_greeks(55.0, 60.0, 0.75, 0.10, 0.30).gamma == pytest.approx(0.0278, abs=HALF_ULP_4DP)


def test_put_theta_haug_chapter_2():
    """Haug, 2nd ed., Chapter 2, theta example: S = 430, X = 405, T = 0.0833, r = 7%, b = 2%, sigma = 20%
    gives put theta -31.1924 per year."""
    theta = black_scholes_greeks(430.0, 405.0, 0.0833, 0.07, 0.20, q=0.05, option_type="put").theta
    assert theta == pytest.approx(-31.1924, abs=HALF_ULP_4DP)


def test_call_rho_haug_chapter_2():
    """Haug, 2nd ed., Chapter 2, rho example: S = 72, X = 75, T = 1, r = b = 9%, sigma = 19% gives call rho 38.7325."""
    assert black_scholes_greeks(72.0, 75.0, 1.0, 0.09, 0.19).rho == pytest.approx(38.7325, abs=HALF_ULP_4DP)
