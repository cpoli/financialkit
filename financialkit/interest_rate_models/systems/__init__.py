"""Interest-rate models: equilibrium and affine short rates, curve-fitted
models and trees, a two-factor Gaussian model, bond options, caps, floors
and swaptions, the HJM framework, Hull-White trinomial trees for callable bonds and the
option-adjusted spread, mortgage pass-throughs with Richard-Roll
prepayments, and the forward market model for backward-looking rates.
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
from financialkit.interest_rate_models.systems.calibration import g2pp_calibrate, hull_white_calibrate, model_implied_vols
from financialkit.interest_rate_models.systems.callable_bonds import callable_bond_price, option_adjusted_spread
from financialkit.interest_rate_models.systems.curve_fitted import (
    HoLeeModel,
    HullWhiteModel,
    black_derman_toy_tree,
    futures_convexity_adjustment,
    hull_white_tree,
)
from financialkit.interest_rate_models.systems.forward_market import fmm_caplet_price, fmm_volatility_decay, simulate_fmm_rate
from financialkit.interest_rate_models.systems.hjm import hjm_drift, simulate_hjm
from financialkit.interest_rate_models.systems.mbs import mbs_price, richard_roll_cpr
from financialkit.interest_rate_models.systems.short_rate import CIRModel, MertonShortRateModel, VasicekModel, feller_condition, monte_carlo_bond_prices
from financialkit.interest_rate_models.systems.two_factor import G2PlusPlusModel, g2pp_swaption

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
    "hull_white_tree",
    "callable_bond_price",
    "option_adjusted_spread",
    "futures_convexity_adjustment",
    "G2PlusPlusModel",
    "zero_coupon_bond_option",
    "coupon_bond_option",
    "swaption_schedule",
    "forward_swap_rate",
    "forward_swap_annuity",
    "jamshidian_swaption",
    "g2pp_swaption",
    "black_swaption_price",
    "cap_floor_price",
    "model_cap_floor_price",
    "model_implied_vols",
    "hull_white_calibrate",
    "g2pp_calibrate",
    "fmm_volatility_decay",
    "fmm_caplet_price",
    "simulate_fmm_rate",
    "richard_roll_cpr",
    "mbs_price",
    "hjm_drift",
    "simulate_hjm",
]
