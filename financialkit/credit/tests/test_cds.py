"""Tests for CDS legs, par spreads, upfronts and the hazard bootstrap."""

import numpy as np
import pytest

from financialkit.credit import HazardCurve, bootstrap_hazard_curve, cds_legs, cds_par_spread, cds_upfront, upfront_to_spread


def test_hull_annual_cds_by_hand():
    """Hull, 10th ed., Section 25.2: hazard 2%, recovery 40%, rate 5%, annual
    payments and defaults at mid-year; the legs are the sums of Tables 25.2-25.4."""
    lam, R, r = 0.02, 0.4, 0.05
    t = np.arange(1, 6)
    Q = np.exp(-lam * t)
    dQ = np.r_[1.0, Q[:-1]] - Q
    pay = np.sum(Q * np.exp(-r * t))
    acc = np.sum(0.5 * dQ * np.exp(-r * (t - 0.5)))
    prot = (1 - R) * np.sum(dQ * np.exp(-r * (t - 0.5)))
    res = cds_legs(HazardCurve.flat(lam), 5.0, r, R, frequency=1, steps_per_year=1)
    assert res.rpv01 == pytest.approx(pay + acc, rel=1e-12)
    assert res.accrued_on_default == pytest.approx(acc, rel=1e-12)
    assert res.protection_leg == pytest.approx(prot, rel=1e-12)
    assert res.par_spread == pytest.approx(prot / (pay + acc))
    assert res.mtm(res.par_spread) == pytest.approx(0.0, abs=1e-15)


def test_credit_triangle_is_continuous_limit():
    lam, R, r = 0.03, 0.35, 0.04
    # With frequent premium and a fine default grid, s -> lambda (1 - R); the error is
    # O((lambda + r) / frequency), about 6e-5 relative here.
    s = cds_par_spread(HazardCurve.flat(lam), 7.0, r, R, frequency=360, steps_per_year=360)
    assert s == pytest.approx(lam * (1 - R), rel=2e-4)


def test_payment_schedule_stub_and_short_maturity():
    res = cds_legs(HazardCurve.flat(0.01), 1.1, 0.02, frequency=4)
    np.testing.assert_allclose(res.payment_times, [0.1, 0.35, 0.6, 0.85, 1.1])
    with pytest.raises(ValueError):
        cds_legs(HazardCurve.flat(0.01), 0.0, 0.02)


def test_upfront_round_trip():
    c = HazardCurve.flat(0.05)
    up = cds_upfront(c, 5.0, 0.01, 0.03)
    par = cds_par_spread(c, 5.0, 0.03)
    assert up > 0  # spread above the 100 bp coupon: the buyer pays upfront
    assert up == pytest.approx((par - 0.01) * cds_legs(c, 5.0, 0.03).rpv01)
    assert upfront_to_spread(up, 0.01, 5.0, 0.03) == pytest.approx(par, rel=1e-9)


def test_bootstrap_reprices_quotes():
    T = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    s = [0.0050, 0.0065, 0.0080, 0.0100, 0.0110, 0.0115]
    c = bootstrap_hazard_curve(T, s, 0.03, 0.4)
    np.testing.assert_allclose([cds_par_spread(c, Ti, 0.03, 0.4) for Ti in T], s, rtol=1e-10)
    assert np.all(c.hazards > 0)
    # Flat quotes give an (almost) flat curve near the credit triangle.
    flat = bootstrap_hazard_curve([1, 3, 5], [0.012] * 3, 0.03, 0.4)
    np.testing.assert_allclose(flat.hazards, flat.hazards[0], rtol=1e-3)
    assert flat.hazards[0] == pytest.approx(0.02, rel=0.01)


def test_bootstrap_rejects_inverted_arbitrage():
    with pytest.raises(ValueError):
        bootstrap_hazard_curve([1.0, 5.0], [0.05, 0.001], 0.03)
