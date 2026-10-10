"""Tests for Redington immunization and the Fong-Vasicek M-squared bound."""

import numpy as np
import pytest

from financialkit.fixed_income import (
    CashFlows,
    ZeroCurve,
    bond_cash_flows,
    fisher_weil_duration,
    fong_vasicek_bound,
    horizon_value,
    linear_forward_shock,
    m_squared,
    redington_immunize,
)


def _liabilities():
    return CashFlows([4.0, 6.0, 8.0], [300.0, 400.0, 300.0])


def test_redington_matches_pv_and_duration_and_surplus_is_local_minimum():
    curve = ZeroCurve([1, 5, 10, 20], [0.03, 0.035, 0.04, 0.042])
    res = redington_immunize(_liabilities(), [bond_cash_flows(1.0, 0.03, 2.0), bond_cash_flows(1.0, 0.05, 15.0)], curve)
    assert res.pv_assets == pytest.approx(res.pv_liabilities, rel=1e-12)
    assert res.duration_assets == pytest.approx(res.duration_liabilities, rel=1e-10)
    assert res.redington_satisfied
    shifts = np.linspace(-0.03, 0.03, 13)
    surplus = res.surplus(shifts)
    assert surplus[6] == pytest.approx(0.0, abs=1e-8)
    assert np.all(surplus >= -1e-8)


def test_redington_fails_when_assets_do_not_straddle():
    """A single zero at the liability duration has zero dispersion: matching it with assets
    inside the liability span leaves asset convexity below liability convexity."""
    L = CashFlows([2.0, 10.0], [500.0, 500.0])
    res = redington_immunize(L, [bond_cash_flows(1.0, 0.0, 5.0), bond_cash_flows(1.0, 0.0, 6.0)], 0.04)
    assert not res.redington_satisfied
    assert res.surplus([0.02])[0] < 0


def test_redington_rejects_wrong_asset_count_and_equal_durations():
    with pytest.raises(ValueError):
        redington_immunize(_liabilities(), [bond_cash_flows(1.0, 0.0, 5.0)], 0.04)
    with pytest.raises(ValueError):
        redington_immunize(_liabilities(), [bond_cash_flows(1.0, 0.0, 5.0), bond_cash_flows(2.0, 0.0, 5.0)], 0.04)


def test_m_squared_bullet_below_barbell_and_bound_holds():
    curve = ZeroCurve([1, 5, 10, 20], [0.03, 0.035, 0.04, 0.042])
    H = 7.0
    bullet = redington_immunize(CashFlows([H], [1000.0]), [bond_cash_flows(1.0, 0.0, 6.0), bond_cash_flows(1.0, 0.0, 8.0)], curve).portfolio
    barbell = redington_immunize(CashFlows([H], [1000.0]), [bond_cash_flows(1.0, 0.0, 1.0), bond_cash_flows(1.0, 0.0, 20.0)], curve).portfolio
    assert fisher_weil_duration(bullet, curve) == pytest.approx(H, rel=1e-10)
    m2_bullet, m2_barbell = m_squared(bullet, H, curve), m_squared(barbell, H, curve)
    assert m2_bullet < m2_barbell
    for slope in (0.001, 0.003):
        for level in (-0.01, 0.0, 0.02):
            shocked = curve.shifted(linear_forward_shock(level, slope, H))
            for port, m2 in ((bullet, m2_bullet), (barbell, m2_barbell)):
                change = horizon_value(port, shocked, H) / horizon_value(port, curve, H) - 1
                assert change >= fong_vasicek_bound(m2, slope) - 1e-12
