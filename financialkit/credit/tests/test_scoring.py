"""Tests for Altman's Z-score and Fisher's linear discriminant."""

import numpy as np
import pytest

from financialkit.credit import ALTMAN_COEFFICIENTS, altman_ratios, altman_z_score, altman_zone, linear_discriminant


def test_z_score_arithmetic():
    """Altman (1968) weights applied to the ratios of a firm worked by hand."""
    x = altman_ratios(170, 300, 60, 380, 2200, 670, 240)
    np.testing.assert_allclose(x, [170 / 670, 300 / 670, 60 / 670, 380 / 240, 2200 / 670])
    assert altman_z_score(170, 300, 60, 380, 2200, 670, 240) == pytest.approx(float(x @ ALTMAN_COEFFICIENTS))
    assert altman_z_score(170, 300, 60, 380, 2200, 670, 240) == pytest.approx(5.46, abs=0.005)


def test_z_score_broadcasts_and_zones():
    z = altman_z_score([10, -50], [20, -80], [5, -10], [100, 10], [150, 60], 200, [100, 180])
    assert z.shape == (2,)
    np.testing.assert_allclose(z, [1.2 * 0.05 + 1.4 * 0.1 + 3.3 * 0.025 + 0.6 * 1.0 + 0.75, 1.2 * -0.25 + 1.4 * -0.4 + 3.3 * -0.05 + 0.6 * 10 / 180 + 0.3])
    assert altman_zone(z).tolist() == ["distress", "distress"]
    assert altman_zone(1.81) == "grey" and altman_zone(1.8) == "distress" and altman_zone(3.0) == "safe"


def test_linear_discriminant_recovers_direction():
    rng = np.random.default_rng(0)
    w_true = np.array([1.0, -2.0, 0.5])
    good = rng.standard_normal((4000, 3)) + 0.5 * w_true
    bad = rng.standard_normal((4000, 3)) - 0.5 * w_true
    w, cutoff = linear_discriminant(good, bad)
    # With identity within-group covariance, Fisher's direction is the difference of means.
    cos = w @ w_true / (np.linalg.norm(w) * np.linalg.norm(w_true))
    assert cos > 0.99
    assert abs(cutoff) < 0.1 * np.linalg.norm(w)
    assert np.mean(good @ w > cutoff) > 0.8 and np.mean(bad @ w < cutoff) > 0.8
