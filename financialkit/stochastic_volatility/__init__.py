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
check, Lee's moment formula for the wings, and an SVI implied-volatility
surface with Gatheral and Jacquier's calendar and butterfly checks that
feeds Dupire's formula, and Gatheral and Jacquier's arbitrage-free SSVI
surface. Rough volatility: Bayer, Friz and Gatheral's rough Bergomi model,
simulated exactly by Cholesky or by Bennedsen, Lunde and Pakkanen's hybrid
scheme, and Gatheral, Jaisson and Rosenbaum's scaling estimate of the
roughness of volatility. Calibration: any
characteristic-function model fitted to a smile across strikes and
maturities by vega-weighted least squares. Simulation: truncated
Euler schemes (Lord-Koekkoek-van Dijk) and Andersen's
quadratic-exponential scheme for Heston.
"""

from financialkit import __version__
from financialkit.stochastic_volatility.core.base import (
    CharacteristicFunctionModel,
    FFTResult,
    FourierCalibration,
    HestonPaths,
    LocalVolSurface,
    RoughBergomiPaths,
    RoughnessEstimate,
    SABRFit,
    SVIFit,
)
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
from financialkit.stochastic_volatility.visualizers.plots import (
    plot_calibration,
    plot_density,
    plot_heston_paths,
    plot_local_vol,
    plot_roughness,
    plot_smile,
)

__all__ = [
    "__version__",
    "CharacteristicFunctionModel",
    "FFTResult",
    "LocalVolSurface",
    "HestonPaths",
    "SABRFit",
    "SVIFit",
    "FourierCalibration",
    "RoughBergomiPaths",
    "RoughnessEstimate",
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
    "plot_smile",
    "plot_local_vol",
    "plot_heston_paths",
    "plot_density",
    "plot_calibration",
    "plot_roughness",
]
