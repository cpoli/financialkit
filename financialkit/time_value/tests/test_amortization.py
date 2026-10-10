"""Tests for amortizing loans and the APR."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.time_value import amortization_schedule, annual_percentage_rate, annuity_factor, effective_annual_rate, loan_payment


def test_mortgage_payment_closed_form():
    """30-year 200,000 mortgage at 6%: P = L i / (1 - (1 + i)^-n) = 1,199.10."""
    i = 0.005
    assert loan_payment(200_000.0, 0.06, 360) == pytest.approx(200_000 * i / (1 - (1 + i) ** -360), rel=DEFAULT_RTOL)
    assert round(loan_payment(200_000.0, 0.06, 360), 2) == 1199.10


def test_schedule_repays_principal_exactly():
    s = amortization_schedule(200_000.0, 0.06, 360)
    assert s.principal.sum() == pytest.approx(200_000.0, rel=DEFAULT_RTOL)
    assert abs(s.balance[-1]) < 1e-6
    np.testing.assert_allclose(s.interest + s.principal, s.payment, rtol=DEFAULT_RTOL)
    assert s.total_interest == pytest.approx(360 * s.payment - 200_000.0, rel=DEFAULT_RTOL)
    # principal portions grow geometrically at (1 + i)
    np.testing.assert_allclose(s.principal[1:] / s.principal[:-1], 1.005, rtol=DEFAULT_RTOL)
    # balance after k payments is the PV of the remaining ones
    assert s.balance[119] == pytest.approx(s.payment * annuity_factor(0.005, 240), rel=DEFAULT_RTOL)


def test_apr_without_fees_is_note_rate():
    p = loan_payment(10_000.0, 0.09, 60)
    assert annual_percentage_rate(10_000.0, p, 60) == pytest.approx(0.09, rel=1e-8)


def test_fees_raise_apr_and_effective_rate_exceeds_apr():
    p = loan_payment(200_000.0, 0.06, 360)
    apr = annual_percentage_rate(200_000.0 - 4_000.0, p, 360)  # two points
    assert 0.06 < apr < 0.065
    assert effective_annual_rate(apr, 12) > apr
    with pytest.raises(ValueError):
        annual_percentage_rate(1000.0, 10.0, 12)
