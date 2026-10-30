"""Tests for PSA prepayment and mortgage pass-through cash flows."""

import numpy as np
import pytest

from financialkit.fixed_income import mbs_cash_flows, price_from_yield, psa_cpr, single_monthly_mortality
from financialkit.time_value import loan_payment


def test_psa_ramp():
    assert psa_cpr([1, 30, 31, 360]) == pytest.approx([0.002, 0.06, 0.06, 0.06])
    assert psa_cpr(5, psa=300) == pytest.approx(0.03)
    assert psa_cpr(30, psa=2000) == 1.0
    with pytest.raises(ValueError):
        psa_cpr(0)
    with pytest.raises(ValueError):
        psa_cpr(5, psa=-1)


def test_smm_round_trips_cpr():
    smm = single_monthly_mortality(0.06)
    assert 1 - (1 - smm) ** 12 == pytest.approx(0.06, rel=1e-14)
    with pytest.raises(ValueError):
        single_monthly_mortality(1.5)


def test_fabozzi_month_one():
    """Fabozzi's $400m pass-through (WAC 8.125%, 7.5% pass-through, WAM 357, 100 PSA): month 1 cash flows."""
    cf = mbs_cash_flows(400e6, 0.08125, 357, 0.075, psa=100, age=3)
    assert cf.smm[0] == pytest.approx(0.00067, abs=5e-6)
    assert cf.interest[0] == pytest.approx(2_500_000, abs=0.5)
    assert cf.scheduled_principal[0] == pytest.approx(267_535, abs=0.5)
    assert cf.prepayment[0] == pytest.approx(267_470, abs=0.5)
    assert cf.cash_flow[0] == pytest.approx(3_035_005, abs=0.5)


@pytest.mark.parametrize("psa", [0, 100, 300])
def test_principal_repays_balance(psa):
    cf = mbs_cash_flows(1e6, 0.06, 360, 0.055, psa=psa)
    assert cf.principal.sum() == pytest.approx(1e6, rel=1e-12)
    assert np.all(cf.balance > 0)


def test_zero_psa_is_level_payment_mortgage():
    cf = mbs_cash_flows(1e5, 0.06, 360, psa=0)
    assert cf.cash_flow == pytest.approx(np.full(360, loan_payment(1e5, 0.06, 360)), rel=1e-10)
    flat = mbs_cash_flows(1.2e3, 0.0, 12, psa=0)
    assert flat.cash_flow == pytest.approx(np.full(12, 100.0))


def test_faster_prepayment_shortens_life_and_cheapens_premium_pool():
    speeds = [50, 100, 200, 400]
    pools = [mbs_cash_flows(100.0, 0.07, 360, 0.065, psa=s) for s in speeds]
    wal = [p.weighted_average_life for p in pools]
    assert np.all(np.diff(wal) < 0)
    prices = [price_from_yield(p.as_cash_flows(), 0.05, compounding=12) for p in pools]
    assert np.all(np.diff(prices) < 0)  # a premium pool loses value when prepaid at par
    discount = [price_from_yield(p.as_cash_flows(), 0.08, compounding=12) for p in pools]
    assert np.all(np.diff(discount) > 0)


def test_mbs_validation():
    with pytest.raises(ValueError):
        mbs_cash_flows(0.0, 0.05)
    with pytest.raises(ValueError):
        mbs_cash_flows(100.0, 0.05, 0)
