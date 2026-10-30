"""Tests for base and compound correlation."""

import numpy as np
import pytest

from financialkit.credit import HazardCurve, base_correlation, base_correlation_spread, compound_correlations, tranche_spread
from financialkit.fixed_income import NelsonSiegelCurve

CURVE = HazardCurve.flat(0.012)
K = np.array([0.03, 0.07, 0.10, 0.15, 0.30])
SKEW = np.array([0.15, 0.25, 0.32, 0.40, 0.55])


def _quotes(skew, n_names=None, r=0.03):
    A, RA = np.r_[0.0, K[:-1]], np.r_[0.0, skew[:-1]]
    return np.array([base_correlation_spread(a, d, ra, rd, CURVE, 5.0, r, n_names=n_names).par_spread for a, d, ra, rd in zip(A, K, RA, skew, strict=True)])


def test_equal_correlations_reduce_to_tranche_spread():
    for a, d in ((0.0, 0.03), (0.03, 0.07), (0.15, 0.3)):
        flat = tranche_spread(a, d, CURVE, 0.3, 5.0, 0.03, n_names=60)
        res = base_correlation_spread(a, d, 0.3, 0.3, CURVE, 5.0, 0.03, n_names=60)
        assert np.isclose(res.par_spread, flat.par_spread, rtol=1e-12)
        np.testing.assert_allclose(res.expected_loss, flat.expected_loss, atol=1e-12)


def test_bootstrap_recovers_a_skew():
    res = base_correlation(K, _quotes(SKEW), CURVE, 5.0, 0.03)
    np.testing.assert_allclose(res.base_correlations, SKEW, atol=1e-8)
    np.testing.assert_allclose(res.spreads, _quotes(SKEW))


def test_bootstrap_recovers_a_skew_on_a_curve_and_a_finite_pool():
    r = NelsonSiegelCurve(0.04, -0.01, 0.0, 2.0)
    res = base_correlation(K[:3], _quotes(SKEW, n_names=125, r=r)[:3], CURVE, 5.0, r, n_names=125)
    np.testing.assert_allclose(res.base_correlations, SKEW[:3], atol=1e-8)


def test_flat_correlation_gives_a_flat_skew():
    quotes = [tranche_spread(a, d, CURVE, 0.2, 5.0, 0.03).par_spread for a, d in zip(np.r_[0.0, K[:-1]], K, strict=True)]
    np.testing.assert_allclose(base_correlation(K, quotes, CURVE, 5.0, 0.03).base_correlations, 0.2, atol=1e-8)


def test_skew_interpolation():
    res = base_correlation(K, _quotes(SKEW), CURVE, 5.0, 0.03)
    assert np.isclose(res.correlation(0.05), 0.2)
    assert np.isclose(res.correlation(0.01), SKEW[0]) and np.isclose(res.correlation(0.5), SKEW[-1])


def test_mezzanine_compound_correlation_is_not_unique():
    """McGinty et al. (2004): the 3-7% spread is hump-shaped in correlation, so a quote has two roots or none."""
    s = _quotes(SKEW)[1]
    roots = compound_correlations(s, 0.03, 0.07, CURVE, 5.0, 0.03)
    assert roots.size == 2
    for rho in roots:
        assert np.isclose(tranche_spread(0.03, 0.07, CURVE, rho, 5.0, 0.03).par_spread, s, rtol=1e-8)
    assert compound_correlations(3 * s, 0.03, 0.07, CURVE, 5.0, 0.03).size == 0


def test_equity_compound_correlation_is_unique():
    s = tranche_spread(0.0, 0.03, CURVE, 0.35, 5.0, 0.03).par_spread
    np.testing.assert_allclose(compound_correlations(s, 0.0, 0.03, CURVE, 5.0, 0.03), [0.35], atol=1e-8)


def test_bad_quotes_raise():
    with pytest.raises(ValueError, match="no base correlation"):
        base_correlation([0.03], [10.0], CURVE, 5.0, 0.03)
    with pytest.raises(ValueError, match="increase"):
        base_correlation([0.07, 0.03], [0.1, 0.01], CURVE, 5.0, 0.03)
    with pytest.raises(ValueError, match="equal length"):
        base_correlation([0.03, 0.07], [0.1], CURVE, 5.0, 0.03)
    with pytest.raises(ValueError, match="attachment"):
        base_correlation_spread(0.07, 0.03, 0.2, 0.2, CURVE, 5.0, 0.03)
