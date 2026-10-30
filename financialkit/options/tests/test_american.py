"""Tests for the Barone-Adesi-Whaley and Bjerksund-Stensland American approximations."""

import pytest

from financialkit.options import CRRTree, barone_adesi_whaley, bjerksund_stensland, black_scholes_price


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


def test_barone_adesi_whaley_call_with_critical_price_far_above_strike():
    """A low dividend yield pushes the early-exercise boundary past 2K, beyond the initial bracket."""
    res = barone_adesi_whaley(100.0, 100.0, 3.0, 0.05, 0.3, q=0.02, option_type="call")
    assert res.critical_price > 200.0
    assert res.price > res.european_price
    tree = CRRTree(1000).price(100.0, 100.0, 3.0, 0.05, 0.3, q=0.02, option_type="call", american=True).price
    # BAW is an approximation; at three years it is within 1% of the converged tree.
    assert res.price == pytest.approx(tree, rel=0.01)


def test_barone_adesi_whaley_rejects_non_positive_inputs():
    with pytest.raises(ValueError, match="sigma must be positive, got 0"):
        barone_adesi_whaley(100.0, 100.0, 1.0, 0.05, 0.0, option_type="put")


def test_bjerksund_stensland_haug_example():
    """Haug, *Complete Guide*, 2nd ed., Section 3.10.2: S = 42, K = 40, T = 0.75, r = 4%, b = -4%, sigma = 35%."""
    res = bjerksund_stensland(42.0, 40.0, 0.75, 0.04, 0.35, q=0.08)
    assert res.price == pytest.approx(5.2704, abs=5e-5)


@pytest.mark.parametrize("option_type,q", [("put", 0.0), ("put", 0.03), ("call", 0.06), ("call", 0.10)])
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
def test_bjerksund_stensland_is_a_lower_bound_close_to_the_tree(option_type, q, S):
    """A flat exercise trigger is a feasible policy, so its value is below the American price."""
    tree = CRRTree(3000).price(S, 100.0, 1.0, 0.08, 0.3, q, option_type, american=True).price
    res = bjerksund_stensland(S, 100.0, 1.0, 0.08, 0.3, q, option_type)
    assert res.european_price - 1e-12 <= res.price <= tree + 2e-3  # 2e-3 covers the tree's discretization error
    # the flat trigger gives up most out of the money: about 2.3% for the one-year put at S = 120
    assert res.price == pytest.approx(tree, rel=0.03, abs=0.05)


def test_bjerksund_stensland_exercises_beyond_the_trigger():
    put = bjerksund_stensland(100.0, 100.0, 1.0, 0.08, 0.25, option_type="put")
    deep = bjerksund_stensland(0.9 * put.critical_price, 100.0, 1.0, 0.08, 0.25, option_type="put")
    assert deep.price == pytest.approx(100.0 - 0.9 * put.critical_price)
    call = bjerksund_stensland(100.0, 100.0, 1.0, 0.05, 0.25, q=0.1)
    assert bjerksund_stensland(1.1 * call.critical_price, 100.0, 1.0, 0.05, 0.25, q=0.1).price == pytest.approx(
        0.1 * call.critical_price + call.critical_price - 100.0
    )


def test_bjerksund_stensland_without_early_exercise_is_european():
    res = bjerksund_stensland(100.0, 95.0, 1.0, 0.05, 0.2)
    assert res.price == res.european_price and res.critical_price == float("inf")
    put = bjerksund_stensland(100.0, 95.0, 1.0, 0.0, 0.2, option_type="put")
    assert put.price == put.european_price and put.critical_price == 0.0


def test_bjerksund_stensland_put_call_transformation():
    """P(S, K, T, r, q) = C(K, S, T, r = q, q = r) (Bjerksund and Stensland 1993, Section 4)."""
    put = bjerksund_stensland(90.0, 100.0, 0.5, 0.06, 0.3, q=0.02, option_type="put").price
    call = bjerksund_stensland(100.0, 90.0, 0.5, 0.02, 0.3, q=0.06).price
    assert put == pytest.approx(call, rel=1e-12)
