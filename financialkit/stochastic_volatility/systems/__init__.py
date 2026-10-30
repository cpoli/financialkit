"""Stochastic-volatility, local-volatility and jump models, Fourier pricers,
smile parametrisations and Heston simulation.
"""

from financialkit.stochastic_volatility.systems.calibration import CALIBRATION_BOUNDS, fourier_calibrate
from financialkit.stochastic_volatility.systems.fourier import (
    BlackScholesModel,
    carr_madan_fft,
    carr_madan_price,
    cos_density,
    cos_price,
    cos_terms,
    lewis_price,
    truncation_range,
)
from financialkit.stochastic_volatility.systems.heston import HestonModel, heston_price, hull_white_mixing_mc, hull_white_price
from financialkit.stochastic_volatility.systems.jumps import BatesModel, KouModel, MertonJumpModel, VarianceGammaModel, merton_jump_price
from financialkit.stochastic_volatility.systems.local_vol import cev_local_vol, cev_price, dupire_local_vol, local_vol_paths
from financialkit.stochastic_volatility.systems.rough import (
    VOLTERRA_SCHEMES,
    RoughBergomiModel,
    fbm_paths,
    hybrid_scheme_variance,
    rough_bergomi_mc_price,
    rough_bergomi_paths,
    volatility_roughness,
    volterra_covariance,
    volterra_paths,
)
from financialkit.stochastic_volatility.systems.simulation import HESTON_SCHEMES, heston_mc_price, heston_paths
from financialkit.stochastic_volatility.systems.smile import durrleman_g, lee_wing_slope, sabr_calibrate, sabr_implied_vol, svi_fit, svi_total_variance
from financialkit.stochastic_volatility.systems.surface import SSVI_PHIS, SSVISurface, SVISurface, ssvi_surface, svi_surface

__all__ = [
    "BlackScholesModel",
    "HestonModel",
    "MertonJumpModel",
    "VarianceGammaModel",
    "KouModel",
    "BatesModel",
    "heston_price",
    "hull_white_price",
    "hull_white_mixing_mc",
    "merton_jump_price",
    "carr_madan_fft",
    "carr_madan_price",
    "lewis_price",
    "cos_price",
    "cos_density",
    "cos_terms",
    "truncation_range",
    "cev_price",
    "cev_local_vol",
    "dupire_local_vol",
    "local_vol_paths",
    "heston_paths",
    "heston_mc_price",
    "HESTON_SCHEMES",
    "sabr_implied_vol",
    "sabr_calibrate",
    "svi_total_variance",
    "svi_fit",
    "durrleman_g",
    "lee_wing_slope",
    "SVISurface",
    "svi_surface",
    "SSVISurface",
    "ssvi_surface",
    "SSVI_PHIS",
    "RoughBergomiModel",
    "volterra_covariance",
    "volterra_paths",
    "hybrid_scheme_variance",
    "VOLTERRA_SCHEMES",
    "rough_bergomi_paths",
    "rough_bergomi_mc_price",
    "fbm_paths",
    "volatility_roughness",
    "fourier_calibrate",
    "CALIBRATION_BOUNDS",
]
