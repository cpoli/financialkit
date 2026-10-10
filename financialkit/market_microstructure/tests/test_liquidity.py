"""Tests for Amihud illiquidity and the Kyle-lambda regression."""

import numpy as np
import pytest

from financialkit.market_microstructure import amihud_illiquidity, kyle_lambda


def test_amihud_by_hand_and_scaling():
    """Amihud (2002), eq. (1): the mean of |r| / dollar volume over trading days."""
    assert amihud_illiquidity([0.02, -0.01], [1e6, 5e5], scale=1.0) == pytest.approx(2e-8)
    r = np.array([0.01, -0.03, 0.02])
    assert amihud_illiquidity(r, [1e6] * 3) == 2 * amihud_illiquidity(r, [2e6] * 3)
    with pytest.raises(ValueError):
        amihud_illiquidity([0.01], [0.0])


def test_kyle_lambda_regression():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(5000)
    fit = kyle_lambda(0.3 * y + 0.1 * rng.standard_normal(5000), y)
    assert abs(fit.coef[1] - 0.3) < 4 * fit.std_errors[1]
