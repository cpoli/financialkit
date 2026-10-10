"""Tests for the Barone-Adesi-Whaley American approximation."""

import pytest

from financialkit.options import CRRTree, barone_adesi_whaley, black_scholes_price


@pytest.mark.parametrize("option_type,q", [("put", 0.0), ("put", 0.03), ("call", 0.06)])
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
def test_close_to_a_fine_tree(option_type, q, S):
    """Barone-Adesi and Whaley (1987), Tables I-IV: errors well under 1% for T <= 0.5."""
    tree = CRRTree(3000).price(S, 100.0, 0.5, 0.08, 0.3, q, option_type, american=True).price
    res = barone_adesi_whaley(S, 100.0, 0.5, 0.08, 0.3, q, option_type)
    assert res.price == pytest.approx(tree, rel=0.01, abs=0.05)  # out of the money the published errors are a few cents
    assert res.price >= res.european_price - 1e-12


def test_call_without_dividends_equals_european():
    res = barone_adesi_whaley(100.0, 95.0, 1.0, 0.05, 0.2)
    assert res.price == res.european_price == pytest.approx(black_scholes_price(100.0, 95.0, 1.0, 0.05, 0.2))
    assert res.critical_price == float("inf")


def test_deep_in_the_money_put_is_exercised():
    res = barone_adesi_whaley(40.0, 100.0, 1.0, 0.08, 0.25, option_type="put")
    assert res.critical_price > 40.0
    assert res.price == pytest.approx(60.0)


def test_value_pastes_smoothly_at_the_critical_price():
    s_star = barone_adesi_whaley(100.0, 100.0, 1.0, 0.08, 0.25, option_type="put").critical_price
    just_above = barone_adesi_whaley(s_star * (1 + 1e-7), 100.0, 1.0, 0.08, 0.25, option_type="put").price
    assert just_above == pytest.approx(100.0 - s_star, abs=1e-4)
