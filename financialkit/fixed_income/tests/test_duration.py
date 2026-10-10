"""Tests for Macaulay/modified/Fisher-Weil duration, convexity and key rate durations."""

import numpy as np
import pytest

from financialkit.fixed_income import (
    ZeroCurve,
    bond_cash_flows,
    bond_risk,
    convexity,
    effective_duration,
    fisher_weil_convexity,
    fisher_weil_duration,
    key_rate_durations,
    key_rate_weights,
    macaulay_duration,
    modified_duration,
    price_from_yield,
)


def test_hull_table_4_6_duration_and_convexity():
    """Hull, 9th ed., Table 4.6 and Sec. 4.9: 3-year 10% semiannual bond at 12% (cont.):
    price 94.213, duration 2.653, convexity 7.570."""
    risk = bond_risk(bond_cash_flows(100.0, 0.10, 3.0), 0.12)
    assert risk.price == pytest.approx(94.213, abs=1e-3)
    assert risk.macaulay_duration == pytest.approx(2.653, abs=1e-3)
    assert risk.convexity == pytest.approx(7.570, abs=1e-3)
    assert risk.modified_duration == pytest.approx(risk.macaulay_duration, rel=1e-12)


@pytest.mark.parametrize("compounding", [None, 1, 2])
def test_modified_duration_and_convexity_match_finite_differences(compounding):
    cf = bond_cash_flows(100.0, 0.07, 12.0)
    y, h = 0.055, 1e-4
    p0, pu, pd = (price_from_yield(cf, y + s, compounding) for s in (0.0, h, -h))
    # central-difference truncation: relative error ~ h^2 * E[t^3] / 6, about 3e-7 here
    assert modified_duration(cf, y, compounding) == pytest.approx((pd - pu) / (2 * h * p0), rel=2e-6)
    # second difference: truncation O(h^2) and rounding O(eps/h^2) both ~1e-8 relative
    assert convexity(cf, y, compounding) == pytest.approx((pu - 2 * p0 + pd) / (h**2 * p0), rel=1e-5)


def test_hicks_modified_duration_relation():
    cf = bond_cash_flows(100.0, 0.06, 8.0)
    for m in (1, 2, 4, 12):
        assert modified_duration(cf, 0.06, m) == pytest.approx(macaulay_duration(cf, 0.06, m) / (1 + 0.06 / m), rel=1e-12)


def test_zero_coupon_duration_is_maturity():
    assert macaulay_duration(bond_cash_flows(100.0, 0.0, 9.5), 0.03, compounding=2) == pytest.approx(9.5)


def test_duration_convexity_price_approximation_improves():
    risk = bond_risk(bond_cash_flows(100.0, 0.05, 20.0), 0.05)
    dy = 0.02
    exact = price_from_yield(bond_cash_flows(100.0, 0.05, 20.0), 0.05 + dy) - risk.price
    assert abs(risk.price_change(dy, order=2) - exact) < 0.2 * abs(risk.price_change(dy, order=1) - exact)
    assert risk.dv01 == pytest.approx(risk.dollar_duration * 1e-4)


def test_fisher_weil_equals_macaulay_on_flat_curve_and_effective_duration():
    cf = bond_cash_flows(100.0, 0.05, 10.0)
    assert fisher_weil_duration(cf, 0.04) == pytest.approx(macaulay_duration(cf, 0.04), rel=1e-12)
    curve = ZeroCurve([1.0, 5.0, 10.0], [0.02, 0.035, 0.04])
    # central-difference truncation: relative error ~ bump^2 * E[t^3] / 6
    assert effective_duration(cf, curve) == pytest.approx(fisher_weil_duration(cf, curve), rel=2e-6)
    assert fisher_weil_convexity(cf, 0.04) == pytest.approx(convexity(cf, 0.04), rel=1e-12)


def test_key_rate_weights_partition_unity():
    w = key_rate_weights(np.linspace(0, 40, 81), [1, 2, 5, 10, 30])
    np.testing.assert_allclose(w.sum(axis=0), 1.0)
    assert np.all(w >= 0)


def test_key_rate_durations_sum_to_parallel_and_locate_cash_flows():
    curve = ZeroCurve([0.5, 2.0, 5.0, 10.0, 30.0], [0.02, 0.025, 0.03, 0.035, 0.04])
    zero = bond_cash_flows(100.0, 0.0, 5.0)
    res = key_rate_durations(zero, curve, [2.0, 5.0, 10.0])
    np.testing.assert_allclose(res.durations, [0.0, 5.0, 0.0], atol=1e-6)
    coupon = key_rate_durations(bond_cash_flows(100.0, 0.06, 10.0), curve, [1.0, 2.0, 5.0, 10.0, 30.0])
    assert coupon.durations.sum() == pytest.approx(coupon.parallel_duration, rel=1e-7)
    assert coupon.durations[-1] == pytest.approx(0.0, abs=1e-10)
