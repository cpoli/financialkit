"""Tests for Treasury bond futures: conversion factors, the cheapest to deliver and the futures price."""

import numpy as np
import pytest

from financialkit.fixed_income import CashFlows, bond_price, cheapest_to_deliver, conversion_factor, treasury_futures_price


def test_hull_conversion_factors():
    """Hull, 9th ed., Sec. 6.2: 10% with 20y 2m left gives 1.4623; 8% with 18y 4m gives 1.2199."""
    assert conversion_factor(0.10, 20 + 2 / 12) == pytest.approx(1.4623, abs=5e-5)
    assert conversion_factor(0.08, 18 + 4 / 12) == pytest.approx(1.2199, abs=5e-5)


def test_conversion_factor_is_price_at_notional_yield_on_coupon_dates():
    c, T = np.array([0.04, 0.06, 0.09]), np.array([15.0, 22.5, 29.5])
    expected = [bond_price(1.0, ci, Ti, 0.06, compounding=2) for ci, Ti in zip(c, T, strict=True)]
    assert np.allclose(conversion_factor(c, T), expected, rtol=1e-12)
    assert conversion_factor(0.06, 25.0) == pytest.approx(1.0, abs=1e-12)


def test_conversion_factor_rounding():
    # 20y 2m rounds to 20y under quarterly rounding, but not monthly
    assert conversion_factor(0.10, 20 + 2 / 12) == conversion_factor(0.10, 20.0)
    assert conversion_factor(0.10, 20 + 2 / 12, round_to_months=1) != conversion_factor(0.10, 20.0)
    with pytest.raises(ValueError):
        conversion_factor(0.05, 10.0, round_to_months=2)
    with pytest.raises(ValueError):
        conversion_factor(0.05, 0.0)


def test_hull_cheapest_to_deliver():
    """Hull, 9th ed., Sec. 6.2: costs 2.69, 1.87, 2.12, so bond 2 is cheapest."""
    basis = cheapest_to_deliver(93.25, [99.50, 143.50, 119.75], [1.0382, 1.5188, 1.2615])
    assert basis.costs == pytest.approx([2.69, 1.87, 2.12], abs=5e-3)
    assert basis.cheapest == 1
    with pytest.raises(ValueError):
        cheapest_to_deliver(93.25, [99.5], [1.0, 1.1])


def test_hull_futures_price():
    """Hull, 9th ed., Sec. 6.2: cash futures 119.711, quoted 114.859, per conversion factor 71.79."""
    coupons = CashFlows([122 / 365, 305 / 365], [6.0, 6.0])
    args = (0.10, 115.0, 6 * 60 / 182, coupons, 270 / 365, 6 * 148 / 183)
    assert treasury_futures_price(*args) == pytest.approx(114.859, abs=1e-3)
    assert treasury_futures_price(*args, 1.6) == pytest.approx(71.79, abs=5e-3)
    with pytest.raises(ValueError):
        treasury_futures_price(0.1, 100.0, 0.0, coupons, 0.0, 0.0)


def test_ctd_switches_with_yield_level():
    """Above the notional yield the longest-duration bond is cheapest; below it, the shortest."""
    c, T = np.array([0.04, 0.06, 0.08]), np.array([30.0, 22.0, 15.0])
    cf = conversion_factor(c, T)
    for y, expected in ((0.08, 0), (0.04, 2)):
        P = np.array([bond_price(100.0, ci, Ti, y, compounding=2) for ci, Ti in zip(c, T, strict=True)])
        assert cheapest_to_deliver(float(np.min(P / cf)), P, cf).cheapest == expected
