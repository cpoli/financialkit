"""Tests for compounding, equivalent rates, the rule of 72 and compound-interest tables."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.time_value import (
    annuity_factor,
    compound_interest_table,
    continuous_rate,
    discount_factor,
    doubling_time,
    effective_annual_rate,
    equivalent_rate,
    future_value,
    rule_of_72,
)


def test_hull_example_4_1_semiannual_to_continuous():
    """Hull, *Options, Futures, and Other Derivatives* (10th ed.), Example 4.1:
    10% with semiannual compounding is 9.758% continuously compounded."""
    assert equivalent_rate(0.10, 2, None) == pytest.approx(2 * np.log(1.05), rel=DEFAULT_RTOL)
    assert round(equivalent_rate(0.10, 2, None), 5) == 0.09758


def test_hull_example_4_2_continuous_to_quarterly():
    """Hull (10th ed.), Example 4.2: 8% continuously compounded is 8.08% with quarterly compounding."""
    assert round(equivalent_rate(0.08, None, 4), 4) == 0.0808


@pytest.mark.parametrize("m_from, m_to", [(1, 12), (2, None), (None, 365), (4, 4)])
def test_equivalent_rates_give_equal_growth(m_from, m_to):
    r = 0.07
    r2 = equivalent_rate(r, m_from, m_to)
    assert future_value(1.0, r2, 3.5, m_to) == pytest.approx(future_value(1.0, r, 3.5, m_from), rel=DEFAULT_RTOL)


def test_bernoulli_limit_is_e():
    """(1 + 1/m)^m increases to e (Bernoulli, 1690), with error ~ e / (2m)."""
    m = np.array([1, 10, 100, 1000, 10_000])
    values = np.array([future_value(1.0, 1.0, 1.0, int(k)) for k in m])
    assert np.all(np.diff(values) > 0)
    assert future_value(1.0, 1.0, 1.0) == pytest.approx(np.e, rel=DEFAULT_RTOL)
    assert np.allclose((np.e - values) * 2 * m / np.e, 1.0, rtol=0.6)


def test_discount_factor_broadcasts_and_inverts_future_value():
    t = np.array([0.5, 1.0, 2.0])
    df = discount_factor(0.04, t, compounding=2)
    assert df.shape == (3,)
    np.testing.assert_allclose(df * future_value(1.0, 0.04, t, 2), 1.0, rtol=DEFAULT_RTOL)


def test_effective_annual_rate_and_continuous_rate():
    assert effective_annual_rate(0.12, 12) == pytest.approx(1.01**12 - 1, rel=DEFAULT_RTOL)
    assert continuous_rate(0.05) == 0.05
    with pytest.raises(ValueError):
        continuous_rate(0.05, 0)


def test_rule_of_72_close_to_exact_doubling_time():
    """Pacioli's rule is within 2% of ln 2 / ln(1 + r) for rates between 4% and 12%."""
    r = np.linspace(0.04, 0.12, 9)
    exact = doubling_time(r, compounding=1)
    np.testing.assert_allclose(rule_of_72(r), exact, rtol=0.02)
    assert rule_of_72(0.08) == 9.0
    assert doubling_time(0.05) == pytest.approx(np.log(2) / 0.05, rel=DEFAULT_RTOL)


def test_compound_interest_table_is_consistent():
    table = compound_interest_table(0.06, 20)
    np.testing.assert_allclose(table.amount * table.discount, 1.0, rtol=DEFAULT_RTOL)
    np.testing.assert_allclose(table.annuity_present, annuity_factor(0.06, table.periods), rtol=DEFAULT_RTOL)
    # s_n = ((1 + i)^n - 1) / i
    np.testing.assert_allclose(table.annuity_future, (table.amount - 1) / 0.06, rtol=DEFAULT_RTOL)
    with pytest.raises(ValueError):
        compound_interest_table(0.06, 0)
