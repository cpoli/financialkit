"""financialkit.interest_rate_models: short-rate models, curve-fitted models, HJM and rate options.

Equilibrium short rates: Merton's arithmetic Brownian rate, Vasicek and
Cox-Ingersoll-Ross (with Feller's condition and exact noncentral
chi-square sampling), and Duffie-Kan affine models of any dimension
priced through their Riccati ODEs (:func:`scipy.integrate.solve_ivp`).
Curve-fitted models take an initial
:class:`~financialkit.fixed_income.core.base.YieldCurve`: Ho-Lee,
Hull-White, the two-factor G2++ and the Black-Derman-Toy lognormal tree
(calibrated by forward induction). Every closed-form bond price is
checked against Feynman-Kac Monte Carlo through
:mod:`financialkit.integrators`.

Options: zero-coupon bond options in any Gaussian model (Jamshidian 1989,
via the :func:`financialkit.options.black76_price` implementation),
coupon bond options and swaptions by Jamshidian's decomposition, caps and
floors by Black's formula on forward rates (the Brace-Gatarek-Musiela
market model) or as bond options, Black swaptions under the swap market
model, and the futures convexity adjustment. The Heath-Jarrow-Morton
framework: the no-arbitrage drift and one-factor forward-curve
simulation, with and without it.
"""

from financialkit import __version__
from financialkit.interest_rate_models.core.base import (
    BondMonteCarloResult,
    CapFloorResult,
    GaussianShortRateModel,
    HJMResult,
    JamshidianResult,
    ModelCurve,
    RateTree,
    ShortRateModel,
)
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
from financialkit.interest_rate_models.utils.numerics import path_discount_factors
from financialkit.interest_rate_models.visualizers.plots import (
    plot_bond_monte_carlo,
    plot_caplets,
    plot_forward_curves,
    plot_jamshidian,
    plot_model_yields,
    plot_rate_tree,
    plot_short_rate_paths,
)

__all__ = [
    "__version__",
    "ShortRateModel",
    "GaussianShortRateModel",
    "ModelCurve",
    "BondMonteCarloResult",
    "RateTree",
    "CapFloorResult",
    "JamshidianResult",
    "HJMResult",
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
    "path_discount_factors",
    "plot_short_rate_paths",
    "plot_model_yields",
    "plot_bond_monte_carlo",
    "plot_rate_tree",
    "plot_caplets",
    "plot_jamshidian",
    "plot_forward_curves",
]
