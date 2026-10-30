"""Tests for Duffie-Kan affine models: Riccati prices against Vasicek, CIR and Monte Carlo."""

import numpy as np
import pytest

from financialkit.interest_rate_models import AffineModel, CIRModel, VasicekModel, monte_carlo_bond_prices

T = np.array([0.5, 2.0, 10.0, 30.0])


def test_riccati_recovers_vasicek():
    assert np.allclose(AffineModel.vasicek(0.03, 0.4, 0.05, 0.015).zero_coupon_bond(T), VasicekModel(0.03, 0.4, 0.05, 0.015).zero_coupon_bond(T), rtol=1e-8)


def test_riccati_recovers_cir():
    assert np.allclose(AffineModel.cir(0.03, 0.4, 0.05, 0.12).zero_coupon_bond(T), CIRModel(0.03, 0.4, 0.05, 0.12).zero_coupon_bond(T), rtol=1e-8)


def test_riccati_recovers_cir_at_a_later_date_and_state():
    a = AffineModel.cir(0.03, 0.4, 0.05, 0.12)
    c = CIRModel(0.03, 0.4, 0.05, 0.12)
    assert a.zero_coupon_bond(9.0, t=2.0, state=[0.07]) == pytest.approx(float(c.zero_coupon_bond(9.0, t=2.0, state=0.07)), rel=1e-8)


def test_independent_factors_multiply_bond_prices():
    """r = x1 + x2 with independent Vasicek and CIR factors: P = P_Vasicek * P_CIR."""
    m = AffineModel(
        x0=[0.01, 0.02],
        K0=[0.5 * 0.01, 0.3 * 0.03],
        K1=[[-0.5, 0.0], [0.0, -0.3]],
        sigma=[0.01, 0.1],
        alpha=[1.0, 0.0],
        beta=[[0.0, 0.0], [0.0, 1.0]],
        delta0=0.0,
        delta1=[1.0, 1.0],
    )
    expected = VasicekModel(0.01, 0.5, 0.01, 0.01).zero_coupon_bond(T) * CIRModel(0.02, 0.3, 0.03, 0.1).zero_coupon_bond(T)
    assert np.allclose(m.zero_coupon_bond(T), expected, rtol=1e-8)
    assert m.r0 == pytest.approx(0.03)
    assert m.riccati(T)[1].shape == (4, 2)


def test_two_factor_affine_bond_matches_monte_carlo():
    m = AffineModel([0.01, 0.02], [0.005, 0.009], [[-0.5, 0.1], [0.0, -0.3]], [0.01, 0.1], [1.0, 0.0], [[0.0, 0.0], [0.0, 1.0]], 0.005, [1.0, 1.0])
    res = monte_carlo_bond_prices(m, [1.0, 4.0], n_steps=160, n_paths=20_000, rng=5)
    assert np.all(np.abs(res.z_scores) < 4)


def test_scalar_and_zero_maturity():
    m = AffineModel.vasicek(0.03, 0.4, 0.05, 0.015)
    assert isinstance(m.zero_coupon_bond(1.0), float)
    assert m.zero_coupon_bond(np.array([0.0, 1.0]))[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        m.riccati([-1.0])


def test_affine_bond_prices_a_batch_of_states_row_by_row():
    m = AffineModel([0.01, 0.02], [0.005, 0.009], [[-0.5, 0.1], [0.0, -0.3]], [0.01, 0.1], [1.0, 0.0], [[0.0, 0.0], [0.0, 1.0]], 0.005, [1.0, 1.0])
    T = np.array([1.0, 3.0, 7.0])
    states = np.array([[0.01, 0.02], [0.03, 0.01], [-0.01, 0.05]])
    batch = m.zero_coupon_bond(T, state=states)
    assert batch.shape == (3, 3)
    for row, x in zip(batch, states, strict=True):
        np.testing.assert_allclose(row, m.zero_coupon_bond(T, state=x), rtol=1e-14)
    assert repr(m) == "AffineModel(n_factors=2, x0=array([0.01, 0.02]))"
