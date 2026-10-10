"""Tests for annuities, perpetuities, life annuities and sinking funds."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.time_value import (
    annuity_factor,
    annuity_future_value,
    annuity_payment,
    annuity_present_value,
    de_witt_survival,
    life_annuity_value,
    perpetuity_present_value,
    sinking_fund_payment,
    sinking_fund_schedule,
)


def test_annuity_factor_closed_form_and_zero_rate():
    i, n = 0.05, 10
    assert annuity_factor(i, n) == pytest.approx((1 - 1.05**-10) / 0.05, rel=DEFAULT_RTOL)
    assert annuity_factor(0.0, n) == n
    assert annuity_factor(i, n, due=True) == pytest.approx(1.05 * annuity_factor(i, n), rel=DEFAULT_RTOL)


def test_annuity_future_value_and_payment_round_trip():
    pv = annuity_present_value(250.0, 0.04, 8)
    assert annuity_future_value(250.0, 0.04, 8) == pytest.approx(pv * 1.04**8, rel=DEFAULT_RTOL)
    assert annuity_payment(pv, 0.04, 8) == pytest.approx(250.0, rel=DEFAULT_RTOL)


def test_perpetuity_is_limit_of_annuity():
    """A consol paying 3 at 3% is worth par, 100; long annuities approach P / i."""
    assert perpetuity_present_value(3.0, 0.03) == pytest.approx(100.0, rel=DEFAULT_RTOL)
    assert annuity_present_value(3.0, 0.03, 5000) == pytest.approx(100.0, rel=DEFAULT_RTOL)
    assert perpetuity_present_value(1.0, 0.08, growth=0.03) == pytest.approx(20.0, rel=DEFAULT_RTOL)
    with pytest.raises(ValueError):
        perpetuity_present_value(1.0, 0.03, growth=0.03)


def test_de_witt_survival_bands():
    """Hazard ratios 1 : 1.5 : 2 : 3, and certain death at 80."""
    h = 0.01
    s = de_witt_survival(50.0, [0.0, 3.0, 13.0, 23.0, 29.0, 30.0], base_hazard=h)
    expected_cum = np.array([0, 3 * h, 3 * h + 10 * 1.5 * h, 3 * h + 15 * h + 20 * h, 3 * h + 15 * h + 20 * h + 18 * h])
    np.testing.assert_allclose(s[:5], np.exp(-expected_cum), rtol=DEFAULT_RTOL)
    assert s[5] == 0.0
    with pytest.raises(ValueError):
        de_witt_survival(1.0, [1.0])


def test_life_annuity_below_annuity_certain():
    k = np.arange(1, 31)
    p = de_witt_survival(50.0, k)
    life = life_annuity_value(1.0, 0.04, p)
    assert 0 < life < annuity_factor(0.04, 30)
    assert life == pytest.approx(np.sum(p * 1.04**-k), rel=DEFAULT_RTOL)
    assert life_annuity_value(1.0, 0.04, np.ones(30)) == pytest.approx(annuity_factor(0.04, 30), rel=DEFAULT_RTOL)


def test_sinking_fund_accumulates_target():
    """Deposits plus interest earned equal the target (Price, 1772)."""
    s = sinking_fund_schedule(1000.0, 0.05, 10)
    assert s.deposit == pytest.approx(sinking_fund_payment(1000.0, 0.05, 10), rel=DEFAULT_RTOL)
    assert s.balance[-1] == pytest.approx(1000.0, rel=DEFAULT_RTOL)
    assert 10 * s.deposit + s.interest.sum() == pytest.approx(1000.0, rel=DEFAULT_RTOL)
    # sinking fund deposit = level loan payment minus interest: 1 / s_n = 1 / a_n - i
    assert sinking_fund_payment(1.0, 0.05, 10) == pytest.approx(1 / annuity_factor(0.05, 10) - 0.05, rel=DEFAULT_RTOL)
