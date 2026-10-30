"""Tests for regression exposure on Hull-White paths and the CVA, DVA and FVA built on it."""

import numpy as np
import pytest

from financialkit.credit import (
    ExposureProfile,
    HazardCurve,
    bilateral_cva,
    funding_valuation_adjustment,
    pathwise_discount,
    regression_mark_to_market,
    swap_cva_swaptions,
    swap_exposure,
    unilateral_cva,
)
from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import HullWhiteModel, VasicekModel, forward_swap_rate, jamshidian_swaption

CURVE = NelsonSiegelCurve(0.04, -0.015, 0.01, 2.0)
HW = HullWhiteModel(CURVE, 0.05, 0.01)
K = forward_swap_rate(CURVE, 0.0, 5.0, 2)
SPP = 4  # exposure dates per semiannual period, swap_exposure's default


@pytest.fixture(scope="module")
def exact():
    return swap_exposure(HW, K, 5.0, n_paths=8000, rng=11, method="exact")


def test_pathwise_discount_reprices_the_curve(exact):
    """Hull-White is fitted to the curve, so E[D(0, t)] = P(0, t) within 4 standard errors."""
    D = exact.discount
    se = D.std(axis=0, ddof=1) / np.sqrt(D.shape[0])
    assert np.all(np.abs(D.mean(axis=0) - CURVE.discount(exact.times)) <= 4 * se + 1e-12)


def test_exact_epe_at_payment_dates_is_a_payer_swaption(exact):
    """After the payment at T_i a payer swap is worth a swap to T_n: E[D V^+] = Jamshidian's swaption."""
    for i in range(1, 10):
        j = i * SPP
        swaption = jamshidian_swaption(HW, i / 2, 5.0 - i / 2, K, 2).price
        assert abs(exact.epe[j] - swaption) < 4 * exact.epe_std_error[j]


def test_exact_ene_is_a_receiver_swaption(exact):
    j = 4 * SPP
    swaption = jamshidian_swaption(HW, 2.0, 3.0, K, 2, payer=False).price
    assert abs(exact.ene[j] - swaption) < 4 * exact.ene_std_error[j]


def test_regression_exposure_is_unbiased():
    """Over 12 independent seeds the regression EPE at payment dates matches the swaptions within 4 standard errors of the mean."""
    swaptions = np.array([jamshidian_swaption(HW, i / 2, 5.0 - i / 2, K, 2).price for i in range(1, 10)])
    diffs = np.array([swap_exposure(HW, K, 5.0, n_paths=2000, rng=seed).epe[SPP:-1:SPP] - swaptions for seed in range(12)])
    se = diffs.std(axis=0, ddof=1) / np.sqrt(diffs.shape[0])
    assert np.all(np.abs(diffs.mean(axis=0)) < 4 * se)


def test_receiver_is_the_negated_payer_on_the_same_paths():
    payer = swap_exposure(HW, K, 3.0, notional=2.0, n_paths=500, rng=3)
    receiver = swap_exposure(HW, K, 3.0, notional=2.0, payer=False, n_paths=500, rng=3)
    np.testing.assert_allclose(receiver.values, -payer.values, atol=1e-14)
    np.testing.assert_allclose(receiver.epe, payer.ene, atol=1e-14)


def test_swap_ends_at_zero_and_works_for_vasicek():
    prof = swap_exposure(VasicekModel(0.03, 0.2, 0.04, 0.01), 0.035, 2.0, freq=4, n_paths=500, rng=0, method="exact")
    assert prof.times.size == 8 * SPP + 1
    assert np.all(prof.values[:, -1] == 0.0)
    assert np.all(prof.pfe(0.99) >= prof.pfe(0.95)) and np.all(prof.pfe(0.95) >= 0.0)
    # rates stay positive on these paths, so discounting shrinks the exposure
    assert np.all(prof.expected_exposure >= prof.epe)


def test_regression_reproduces_a_deterministic_value():
    t = np.linspace(0.0, 3.0, 7)
    D = np.exp(-0.04 * t) * np.ones((50, 1))
    C = np.zeros((50, 7))
    C[:, 2::2] = 0.5
    V = regression_mark_to_market(C, D, np.random.default_rng(0).normal(size=(50, 7, 2)))
    expected = [sum(0.5 * np.exp(-0.04 * (t[j] - t[i])) for j in (2, 4, 6) if j > i) for i in range(7)]
    np.testing.assert_allclose(V, np.tile(expected, (50, 1)), atol=1e-12)


def test_pathwise_discount_is_exact_for_linear_rates():
    t = np.linspace(0.0, 2.0, 9)
    np.testing.assert_allclose(pathwise_discount(t, [0.01 + 0.02 * t]), [np.exp(-(0.01 * t + 0.01 * t**2))], atol=1e-14)


def test_unilateral_cva_matches_the_swaption_strip(exact):
    """Brigo-Masetti: on the payment dates, CVA from the exposure profile equals the strip of swaptions."""
    cpty = HazardCurve([1.0, 3.0, 5.0], [0.01, 0.02, 0.03])
    on_payments = ExposureProfile(exact.times[::SPP], exact.values[:, ::SPP], exact.discount[:, ::SPP], "exact")
    cva = unilateral_cva(on_payments, cpty, 0.35)
    closed = swap_cva_swaptions(HW, K, 5.0, cpty, 0.35)
    per_path = 0.65 * (on_payments.discount * np.maximum(on_payments.values, 0.0))[:, 1:] @ -np.diff(cpty.survival(on_payments.times))
    assert np.isclose(cva, per_path.mean())
    assert abs(cva - closed) < 4 * per_path.std(ddof=1) / np.sqrt(per_path.size)
    # between payments the exposure still holds the coming coupon, which the strip's post-payment dates miss
    assert closed < unilateral_cva(exact, cpty, 0.35) < 1.2 * closed


def test_bilateral_prices_agree_across_the_two_sides():
    """Gregory (2009): with first-to-default, our DVA is the counterparty's CVA and the two adjustments cancel."""
    us, them = HazardCurve.flat(0.01), HazardCurve.flat(0.03)
    ours = swap_exposure(HW, K, 5.0, n_paths=2000, rng=5)
    theirs = swap_exposure(HW, K, 5.0, payer=False, n_paths=2000, rng=5)
    a = bilateral_cva(ours, them, us, 0.4, 0.3)
    b = bilateral_cva(theirs, us, them, 0.3, 0.4)
    assert np.isclose(a.cva, b.dva, rtol=1e-12) and np.isclose(a.dva, b.cva, rtol=1e-12)
    assert np.isclose(a.bva, -b.bva, rtol=1e-12)
    naive = bilateral_cva(ours, them, us, 0.4, 0.3, first_to_default=False)
    assert naive.cva > a.cva and naive.dva > a.dva and not naive.first_to_default
    # without own default risk the bilateral CVA is the unilateral one
    assert np.isclose(bilateral_cva(ours, them, HazardCurve.flat(0.0)).cva, unilateral_cva(ours, them))


def test_funding_benefit_at_own_spread_is_the_dva():
    """Burgard-Kjaer: lending at s = (1 - R_B) lambda_B gives an FBA equal to the DVA, up to the bucket rule.

    With flat hazards the DVA bucket is Q(t_i)(e^{lambda dt} - 1) against lambda dt Q(t_i) for the FBA,
    a relative gap of about lambda dt / 2 = 0.25% at dt = 0.125.
    """
    us, them = HazardCurve.flat(0.04), HazardCurve.flat(0.02)
    prof = swap_exposure(HW, K, 5.0, n_paths=2000, rng=6)
    dva = bilateral_cva(prof, them, us, recovery_own=0.4).dva
    fva = funding_valuation_adjustment(prof, 0.6 * 0.04, counterparty=them, own=us)
    assert abs(fva.fba / dva - 1) < 0.003
    assert fva.fva == fva.fca - fva.fba
    asymmetric = funding_valuation_adjustment(prof, 0.6 * 0.04, lending_spread=0.0, counterparty=them, own=us)
    assert asymmetric.fba == 0.0 and asymmetric.fca == fva.fca


def test_fva_with_a_spread_term_structure():
    t = np.linspace(0.0, 2.0, 5)
    prof = ExposureProfile(t, np.full((3, 5), 2.0), np.ones((3, 5)), "exact")
    res = funding_valuation_adjustment(prof, np.array([0.0, 0.01, 0.01, 0.02, 0.02]))
    assert np.isclose(res.fca, 2.0 * 0.5 * (0.01 + 0.01 + 0.02 + 0.02))


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (swap_exposure, {"model": HW, "fixed_rate": 0.03, "maturity": 1.3}),
        (swap_exposure, {"model": HW, "fixed_rate": 0.03, "maturity": 1.0, "method": "closed"}),
        (swap_cva_swaptions, {"model": HW, "fixed_rate": 0.03, "maturity": 1.3, "counterparty": HazardCurve.flat(0.01)}),
        (swap_cva_swaptions, {"model": HW, "fixed_rate": 0.03, "maturity": 1.0, "counterparty": HazardCurve.flat(0.01), "recovery": 1.5}),
        (regression_mark_to_market, {"cash_flows": np.zeros((3, 4)), "discount": np.ones((3, 5)), "states": np.zeros((3, 4))}),
    ],
)
def test_bad_input_raises(func, kwargs):
    with pytest.raises(ValueError):
        func(**kwargs)
