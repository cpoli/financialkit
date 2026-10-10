"""financialkit.stochastic_volatility: beyond constant volatility.

Local volatility: Cox's constant elasticity of variance with Schroder's
noncentral chi-square prices, and Dupire's local volatility recovered
from a price surface and simulated through
:func:`financialkit.integrators.euler_maruyama`. Stochastic volatility:
Hull and White's mixing formula, and the Heston model with its
two-probability formula. Jumps: Merton's jump diffusion (series formula),
the Madan-Seneta and Madan-Carr-Chang variance gamma process, Kou's
double-exponential jumps and Bates's Heston-with-jumps. Fourier
pricing for every characteristic-function model: Carr-Madan's FFT,
Lewis's single integral and Fang-Oosterlee's COS method. Smiles: Hagan's
SABR formula and calibration, Gatheral's SVI with Durrleman's arbitrage
check, and Lee's moment formula for the wings. Simulation: truncated
Euler schemes (Lord-Koekkoek-van Dijk) and Andersen's
quadratic-exponential scheme for Heston.
"""

from financialkit import __version__
from financialkit.stochastic_volatility.core.base import CharacteristicFunctionModel, FFTResult, HestonPaths, LocalVolSurface, SABRFit, SVIFit
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
from financialkit.stochastic_volatility.systems.simulation import HESTON_SCHEMES, heston_mc_price, heston_paths
from financialkit.stochastic_volatility.systems.smile import durrleman_g, lee_wing_slope, sabr_calibrate, sabr_implied_vol, svi_fit, svi_total_variance

__all__ = [
    "__version__",
    "CharacteristicFunctionModel",
    "FFTResult",
    "LocalVolSurface",
    "HestonPaths",
    "SABRFit",
    "SVIFit",
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
]
