"""Tests that every credit pricer takes a ``fixed_income`` discount curve as well as a flat rate."""

import numpy as np
import pytest
from scipy import integrate

from financialkit.credit import (
    HazardCurve,
    bootstrap_hazard_curve,
    cds_legs,
    cds_par_spread,
    cds_upfront,
    duffie_singleton_cir_zero,
    jarrow_turnbull_zero,
    jlt_risky_zero,
    kmv_iterate,
    merton_calibrate,
    merton_model,
    risky_zero_coupon,
    tranche_spread,
    upfront_to_spread,
)
from financialkit.fixed_income import FlatCurve, ZeroCurve, bootstrap_par_curve

R0 = 0.03
HAZ = HazardCurve([1.0, 3.0, 5.0], [0.01, 0.02, 0.03])
GEN = np.array([[-0.1, 0.08, 0.02], [0.05, -0.15, 0.1], [0.0, 0.0, 0.0]])
# An upward-sloping Treasury curve bootstrapped from par yields.
PAR = bootstrap_par_curve([1, 2, 3, 5, 7, 10], [0.02, 0.025, 0.03, 0.035, 0.038, 0.04])


def _equity():
    rng = np.random.default_rng(0)
    return 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 60)))


# Values the flat-rate functions returned before they took curves (0.1.0).
FLAT_CASES = [
    (lambda r: cds_par_spread(HAZ, 5.0, r), 0.012936372636969683),
    (lambda r: cds_upfront(HAZ, 5.0, 0.01, r), 0.01301015891146913),
    (lambda r: upfront_to_spread(0.02, 0.01, 5.0, r), 0.014583767747815224),
    (lambda r: bootstrap_hazard_curve([1, 3, 5], [0.006, 0.009, 0.012], r).hazards[2], 0.0284239965085774),
    (lambda r: jarrow_turnbull_zero(5.0, r, HAZ, 0.4), 0.8069141420521629),
    (lambda r: risky_zero_coupon(5.0, r, 0.02, 0.4, "rmv"), 0.8105842459701871),
    (lambda r: risky_zero_coupon(5.0, r, 0.02, 0.4, "rt"), 0.8115636604128661),
    (lambda r: risky_zero_coupon(5.0, r, 0.02, 0.4, "rfv"), 0.8141926577799801),
    (lambda r: duffie_singleton_cir_zero(5.0, r, 0.02, 0.5, 0.03, 0.1, 0.4), 0.7956695086064041),
    (lambda r: jlt_risky_zero(5.0, r, GEN, 0, 0.4), 0.7847752160590477),
    (lambda r: tranche_spread(0.03, 0.07, HAZ, 0.3, 5.0, r).par_spread, 0.10249563568335222),
    (lambda r: merton_model(120.0, 100.0, 2.0, r, 0.25).credit_spread, 0.02984130966122938),
    (lambda r: merton_calibrate(30.0, 0.6, 100.0, 2.0, r).asset_value, 122.59157002560742),
    (lambda r: kmv_iterate(_equity(), 40.0, r).asset_volatility, 0.16358107824036344),
]


@pytest.mark.parametrize(("f", "expected"), FLAT_CASES)
def test_flat_curve_reproduces_flat_rate(f, expected):
    # 1e-12: the curve path rewrites e^{-(r + a)T} as e^{-rT} e^{-aT}, a few ulps apart.
    assert f(R0) == pytest.approx(expected, rel=1e-12)
    assert f(FlatCurve(R0)) == pytest.approx(expected, rel=1e-12)


def test_cds_legs_with_curve_match_quadrature():
    """With a fine default grid the legs converge to the integrals
    (1 - R) int Z lambda Q du and int Z Q du (O'Kane and Turnbull 2003)."""
    R = 0.4
    res = cds_legs(HAZ, 5.0, PAR, R, freq=360, steps_per_year=360)
    knots = [1.0, 2.0, 3.0]

    def dens(u):
        return PAR.discount(u) * HAZ.hazard(u) * HAZ.survival(u)

    prot = (1 - R) * integrate.quad(dens, 0.0, 5.0, points=knots, epsabs=1e-14)[0]
    annuity = integrate.quad(lambda u: PAR.discount(u) * HAZ.survival(u), 0.0, 5.0, points=knots, epsabs=1e-14)[0]
    # Midpoint defaults: O(h^2) on the protection leg; discrete premium: O(h) on the annuity.
    assert res.protection_leg == pytest.approx(prot, rel=1e-6)
    assert res.rpv01 == pytest.approx(annuity, rel=1e-4)


def test_upward_curve_moves_par_spread_toward_early_hazards():
    """The par spread is a Z(u)Q(u)-weighted average of the hazard rate. Against a
    flat rate at the curve's short end, an upward-sloping curve discounts later
    periods harder, so a rising hazard curve quotes a lower par spread and a
    falling one a higher spread, while a flat hazard barely moves."""
    flat = FlatCurve(float(PAR.zero_rate(0.0)))
    rising = HazardCurve([1.0, 3.0, 5.0], [0.005, 0.02, 0.06])
    falling = HazardCurve([1.0, 3.0, 5.0], [0.06, 0.02, 0.005])
    assert cds_par_spread(rising, 5.0, PAR) < cds_par_spread(rising, 5.0, flat) - 1e-5
    assert cds_par_spread(falling, 5.0, PAR) > cds_par_spread(falling, 5.0, flat) + 1e-5
    h = HazardCurve.flat(0.02)
    # Only the quarterly premium dates make it depend on rates at all: about 0.2% here,
    # against the 1.3-1.8% moves above.
    assert cds_par_spread(h, 5.0, PAR) == pytest.approx(cds_par_spread(h, 5.0, flat), rel=5e-3)


def test_bootstrap_on_curve_reprices_on_that_curve_only():
    T = [1.0, 3.0, 5.0, 7.0, 10.0]
    s = [0.006, 0.009, 0.012, 0.0135, 0.0145]
    on_curve = bootstrap_hazard_curve(T, s, PAR)
    np.testing.assert_allclose([cds_par_spread(on_curve, Ti, PAR) for Ti in T], s, rtol=1e-10)
    flat = bootstrap_hazard_curve(T, s, float(PAR.zero_rate(10.0)))
    assert np.max(np.abs(on_curve.hazards - flat.hazards)) > 1e-5
    with pytest.raises(AssertionError):
        np.testing.assert_allclose([cds_par_spread(on_curve, Ti, flat.hazards[0]) for Ti in T], s, rtol=1e-6)


def test_risky_zeros_scale_with_curve_discount_factor():
    """The riskless factor P(0, T) multiplies the default-adjusted one, so a curve
    gives the flat-rate answer at the zero rate to each maturity."""
    T = np.array([0.5, 2.0, 5.0, 8.0])
    y = PAR.zero_rate(T)
    np.testing.assert_allclose(jarrow_turnbull_zero(T, PAR, HAZ, 0.4), jarrow_turnbull_zero(T, y, HAZ, 0.4), rtol=1e-12)
    for conv in ("rmv", "rt"):
        np.testing.assert_allclose(risky_zero_coupon(T, PAR, 0.02, 0.4, conv), risky_zero_coupon(T, y, 0.02, 0.4, conv), rtol=1e-12)
    dsc = [duffie_singleton_cir_zero(t, float(yt), 0.02, 0.5, 0.03, 0.1, 0.4) for t, yt in zip(T, y, strict=True)]
    np.testing.assert_allclose(duffie_singleton_cir_zero(T, PAR, 0.02, 0.5, 0.03, 0.1, 0.4), dsc, rtol=1e-12)
    jlt = [jlt_risky_zero(t, float(yt), GEN, 1, 0.4) for t, yt in zip(T, y, strict=True)]
    np.testing.assert_allclose(jlt_risky_zero(T, PAR, GEN, 1, 0.4), jlt, rtol=1e-12)


def test_recovery_of_face_value_on_curve_matches_quadrature():
    T, h, R = 6.0, 0.03, 0.4
    annuity = integrate.quad(lambda u: PAR.discount(u) * np.exp(-h * u), 0.0, T, points=[1, 2, 3, 5], epsabs=1e-14)[0]
    expected = PAR.discount(T) * np.exp(-h * T) + R * h * annuity
    assert risky_zero_coupon(T, PAR, h, R, "rfv") == pytest.approx(expected, rel=1e-10)
    # Recovery paid at default is worth more than recovery of treasury when rates are positive.
    assert risky_zero_coupon(T, PAR, h, R, "rfv") > risky_zero_coupon(T, PAR, h, R, "rt")


def test_tranche_spread_on_curve():
    curve = ZeroCurve([1.0, 5.0], [0.01, 0.06])
    res = tranche_spread(0.03, 0.07, HAZ, 0.3, 5.0, curve)
    assert res.par_spread != pytest.approx(tranche_spread(0.03, 0.07, HAZ, 0.3, 5.0, R0).par_spread, rel=1e-4)
    # Discounting is the same for both legs' expected losses, so a parallel shift barely moves it.
    shifted = tranche_spread(0.03, 0.07, HAZ, 0.3, 5.0, curve.shifted(0.01))
    assert shifted.par_spread == pytest.approx(res.par_spread, rel=1e-2)


def test_structural_models_use_zero_rate_to_maturity():
    """With deterministic rates the debt sees the curve only through P(0, T)."""
    curve = ZeroCurve([1.0, 2.0, 5.0], [0.02, 0.035, 0.05])
    y = float(curve.zero_rate(2.0))
    a, b = merton_model(120.0, 100.0, 2.0, curve, 0.25), merton_model(120.0, 100.0, 2.0, y, 0.25)
    assert a == b
    assert merton_calibrate(30.0, 0.6, 100.0, 2.0, curve) == merton_calibrate(30.0, 0.6, 100.0, 2.0, y)
    E = _equity()
    assert kmv_iterate(E, 40.0, curve, T=2.0).asset_volatility == kmv_iterate(E, 40.0, y, T=2.0).asset_volatility
