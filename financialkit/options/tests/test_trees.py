"""Tests for the CRR, Rendleman-Bartter, Leisen-Reimer and Boyle trinomial lattices."""

import numpy as np
import pytest

from financialkit.options import CRRTree, LeisenReimerTree, RendlemanBartterTree, TrinomialTree, black_scholes_greeks, black_scholes_price

TREES = [CRRTree, RendlemanBartterTree, LeisenReimerTree, TrinomialTree]


def test_crr_hull_example_21_1():
    """Hull, 10th ed., Example 21.1 / Figure 21.3: five-step American put worth 4.49."""
    res = CRRTree(5).price(50.0, 50.0, 5 / 12, 0.10, 0.40, option_type="put", american=True)
    assert res.price == pytest.approx(4.49, abs=5e-3)
    assert res.exercise_boundary.shape == (6,)


@pytest.mark.parametrize("tree", TREES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_prices_converge_to_black_scholes(tree, option_type):
    exact = black_scholes_price(100.0, 95.0, 1.0, 0.05, 0.25, 0.02, option_type)
    res = tree(400).price(100.0, 95.0, 1.0, 0.05, 0.25, 0.02, option_type)
    # CRR-type error is O(1/n) with constant < 5 here
    assert res.price == pytest.approx(exact, abs=0.0125)
    g = black_scholes_greeks(100.0, 95.0, 1.0, 0.05, 0.25, 0.02, option_type)
    assert res.delta == pytest.approx(g.delta, abs=5e-3)
    assert res.gamma == pytest.approx(g.gamma, abs=5e-4)


def test_leisen_reimer_converges_at_second_order():
    exact = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)
    e1 = abs(LeisenReimerTree(51).price(100.0, 100.0, 1.0, 0.05, 0.2).price - exact)
    e2 = abs(LeisenReimerTree(201).price(100.0, 100.0, 1.0, 0.05, 0.2).price - exact)
    order = np.log(e1 / e2) / np.log(201 / 51)
    assert order == pytest.approx(2.0, abs=0.15)


def test_leisen_reimer_rounds_even_steps_up():
    assert LeisenReimerTree(100).price(100.0, 100.0, 1.0, 0.05, 0.2).n_steps == 101


def test_rendleman_bartter_probabilities_are_near_one_half():
    tree = RendlemanBartterTree(100)
    u, d, p, _ = tree._parameters(100.0, 100.0, 1.0, 0.05, 0.3, 0.0, 100)
    assert p == pytest.approx(0.5, abs=1e-3)


def test_trinomial_probabilities_match_two_moments():
    tree = TrinomialTree(50)
    T, r, q, sigma = 1.0, 0.05, 0.01, 0.3
    u, pu, pm, pd = tree.probabilities(T, r, sigma, q)
    dt = T / 50
    M = np.exp((r - q) * dt)
    assert pu + pm + pd == pytest.approx(1.0)
    assert pu * u + pm + pd / u == pytest.approx(M)
    assert pu * u**2 + pm + pd / u**2 == pytest.approx(M**2 * np.exp(sigma**2 * dt))


@pytest.mark.parametrize("tree", TREES)
def test_american_put_agrees_across_lattices(tree):
    reference = CRRTree(4000).price(100.0, 100.0, 1.0, 0.08, 0.25, option_type="put", american=True).price
    res = tree(500).price(100.0, 100.0, 1.0, 0.08, 0.25, option_type="put", american=True)
    assert res.price == pytest.approx(reference, abs=5e-3)
    assert res.price > black_scholes_price(100.0, 100.0, 1.0, 0.08, 0.25, option_type="put")


def test_american_call_without_dividends_is_european():
    eu = CRRTree(300).price(100.0, 100.0, 1.0, 0.05, 0.2)
    am = CRRTree(300).price(100.0, 100.0, 1.0, 0.05, 0.2, american=True)
    assert am.price == pytest.approx(eu.price, abs=1e-12)
    assert np.all(np.isnan(am.exercise_boundary[:-1]))


def test_put_exercise_boundary_rises_towards_the_strike():
    b = CRRTree(200).price(100.0, 100.0, 1.0, 0.08, 0.25, option_type="put", american=True).exercise_boundary
    valid = b[~np.isnan(b)]
    assert np.all(valid < 100.0) and valid[-1] > valid[0]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        CRRTree(0)
    with pytest.raises(ValueError):
        CRRTree(10).price(100.0, 100.0, -1.0, 0.05, 0.2)
    with pytest.raises(ValueError):
        TrinomialTree(10, stretch=0.9)


def test_trees_reject_parameters_without_valid_probabilities():
    """A drift that outruns sigma sqrt(dt) leaves CRR's p above one; a stretch near one makes Boyle's middle branch negative."""
    with pytest.raises(ValueError, match="outside"):
        CRRTree(2).price(100.0, 100.0, 1.0, 0.5, 0.05)
    with pytest.raises(ValueError, match="negative branch probability"):
        TrinomialTree(4, stretch=1.01).price(100.0, 100.0, 1.0, 0.5, 0.1)
