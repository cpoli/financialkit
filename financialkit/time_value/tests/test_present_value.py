"""Tests for present value, NPV and Fisher's real rate."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.time_value import annuity_present_value, fisher_nominal_rate, fisher_real_rate, npv, present_value


def test_brealey_myers_office_block():
    """Brealey, Myers and Allen, *Principles of Corporate Finance* (13th ed.), Sec. 2.1:
    an office block costing 370,000 that sells for 420,000 a year later is
    worth 400,000 at 5%, an NPV of 30,000."""
    assert present_value([420_000], [1.0], 0.05, compounding=1) == pytest.approx(400_000, rel=DEFAULT_RTOL)
    assert npv(0.05, [-370_000, 420_000]) == pytest.approx(30_000, rel=DEFAULT_RTOL)


def test_npv_of_level_stream_matches_annuity():
    cf = np.r_[0.0, np.full(10, 50.0)]
    assert npv(0.07, cf) == pytest.approx(annuity_present_value(50.0, 0.07, 10), rel=DEFAULT_RTOL)


def test_present_value_vectorised_over_rates():
    rates = np.array([0.0, 0.05, 0.10])
    pv = present_value([100.0, 100.0], [1.0, 2.0], rates)
    assert pv.shape == (3,)
    assert pv[0] == pytest.approx(200.0)
    assert np.all(np.diff(pv) < 0)
    assert pv[1] == pytest.approx(100 * (np.exp(-0.05) + np.exp(-0.10)), rel=DEFAULT_RTOL)


def test_present_value_rejects_mismatched_times():
    with pytest.raises(ValueError):
        present_value([1.0, 2.0], [1.0], 0.05)


def test_fisher_relation_round_trip_and_approximation():
    """(1 + i) = (1 + r)(1 + pi); the approximation i - pi errs by r * pi / (1 + pi)."""
    i, p = 0.08, 0.03
    r = fisher_real_rate(i, p)
    assert fisher_nominal_rate(r, p) == pytest.approx(i, rel=DEFAULT_RTOL)
    assert fisher_real_rate(i, p, exact=False) - r == pytest.approx(r * p, rel=DEFAULT_RTOL)
    np.testing.assert_allclose(fisher_real_rate([0.05, 0.10], 0.0), [0.05, 0.10])
