"""Interest-rate models: equilibrium and affine short rates, curve-fitted
models and trees, a two-factor Gaussian model, bond options, caps, floors
and swaptions, and the HJM framework.
"""

from financialkit.interest_rate_models.systems.affine import AffineModel
from financialkit.interest_rate_models.systems.bond_options import (
    black_swaption_price,
    cap_floor_price,
    coupon_bond_option,
    forward_swap_annuity,
    forward_swap_rate,
    jamshidian_swaption,
    model_cap_floor_price,
    swaption_schedule,
    zero_coupon_bond_option,
)
from financialkit.interest_rate_models.systems.curve_fitted import HoLeeModel, HullWhiteModel, black_derman_toy_tree, futures_convexity_adjustment
from financialkit.interest_rate_models.systems.hjm import hjm_drift, simulate_hjm
from financialkit.interest_rate_models.systems.short_rate import CIRModel, MertonShortRateModel, VasicekModel, feller_condition, monte_carlo_bond_prices
from financialkit.interest_rate_models.systems.two_factor import G2PlusPlusModel

__all__ = [
    "monte_carlo_bond_prices",
    "feller_condition",
    "MertonShortRateModel",
    "VasicekModel",
    "CIRModel",
    "AffineModel",
    "HoLeeModel",
    "HullWhiteModel",
    "black_derman_toy_tree",
    "futures_convexity_adjustment",
    "G2PlusPlusModel",
    "zero_coupon_bond_option",
    "coupon_bond_option",
    "swaption_schedule",
    "forward_swap_rate",
    "forward_swap_annuity",
    "jamshidian_swaption",
    "black_swaption_price",
    "cap_floor_price",
    "model_cap_floor_price",
    "hjm_drift",
    "simulate_hjm",
]
