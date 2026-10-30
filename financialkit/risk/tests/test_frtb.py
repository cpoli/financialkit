"""Tests for the FRTB liquidity-horizon expected shortfall."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.risk import LIQUIDITY_HORIZONS, expected_shortfall, frtb_es_cascade, frtb_expected_shortfall

X = np.random.default_rng(0).standard_normal((4000, 3))
W = np.array([1.0, 2.0, 3.0])


def linear(dX):
    return dX @ W


def test_cascade_by_hand():
    """Basel (2016), paragraph 181: ES^2 = ES_1^2 + sum_j (ES_j sqrt((LH_j - LH_{j-1}) / 10))^2."""
    e = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    by_hand = np.sqrt(25 + 16 * 1 + 9 * 2 + 4 * 2 + 1 * 6)
    assert frtb_es_cascade(e) == pytest.approx(by_hand, rel=DEFAULT_RTOL)
    with pytest.raises(ValueError, match="liquidity horizon"):
        frtb_es_cascade([1.0, 2.0])


@pytest.mark.parametrize("h", LIQUIDITY_HORIZONS)
def test_one_horizon_for_every_factor_is_the_square_root_of_time(h):
    res = frtb_expected_shortfall(linear, X, [h] * 3)
    assert res.es == pytest.approx(expected_shortfall(linear(X), 0.975) * np.sqrt(h / 10), rel=DEFAULT_RTOL)
    assert np.all(res.es_by_horizon[np.asarray(LIQUIDITY_HORIZONS) > h] == 0)


def test_mixed_horizons_lie_between_the_extremes():
    res = frtb_expected_shortfall(linear, X, [10, 40, 120])
    base = expected_shortfall(linear(X), 0.975)
    assert res.es_by_horizon[0] == pytest.approx(base)
    assert res.es_by_horizon[2] == pytest.approx(expected_shortfall(linear(X * [0, 1, 1]), 0.975))
    assert res.es_by_horizon[4] == pytest.approx(expected_shortfall(3 * X[:, 2], 0.975))
    assert base < res.es < base * np.sqrt(12)
    assert res.es == pytest.approx(np.linalg.norm(res.scaled))


def test_invalid_inputs():
    with pytest.raises(ValueError, match="must be in"):
        frtb_expected_shortfall(linear, X, [10, 30, 120])
    with pytest.raises(ValueError, match="one entry per factor"):
        frtb_expected_shortfall(linear, X, [10, 20])
    with pytest.raises(ValueError, match="one value per scenario"):
        frtb_expected_shortfall(lambda d: d[:10, 0], X, [10, 20, 40])
