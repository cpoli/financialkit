"""Finite-difference pricers: theta-schemes, American complementarity
solvers, grids, stability, the heat-equation transform, the log-spot
trinomial equivalence, Richardson extrapolation, two-factor ADI and the
Heston PDE.
"""

from financialkit.pde_methods.systems.accuracy import richardson_extrapolation
from financialkit.pde_methods.systems.adi import CraigSneydADI, PeacemanRachfordADI
from financialkit.pde_methods.systems.american import brennan_schwartz, penalty_solve, psor
from financialkit.pde_methods.systems.grids import default_s_max, tavella_randall_grid, uniform_grid
from financialkit.pde_methods.systems.heat import heat_equation_exponents, heat_equation_price
from financialkit.pde_methods.systems.heston_adi import HestonADI, heston_spot_grid, heston_variance_grid
from financialkit.pde_methods.systems.log_explicit import explicit_trinomial_probabilities, log_explicit_price
from financialkit.pde_methods.systems.schemes import CrankNicolson, ExplicitScheme, ImplicitScheme, ThetaScheme
from financialkit.pde_methods.systems.stability import diffusion_number, explicit_stability, von_neumann_amplification
from financialkit.pde_methods.systems.tridiagonal import thomas_solve

__all__ = [
    "ThetaScheme",
    "ExplicitScheme",
    "ImplicitScheme",
    "CrankNicolson",
    "thomas_solve",
    "psor",
    "brennan_schwartz",
    "penalty_solve",
    "uniform_grid",
    "tavella_randall_grid",
    "default_s_max",
    "explicit_stability",
    "diffusion_number",
    "von_neumann_amplification",
    "heat_equation_exponents",
    "heat_equation_price",
    "explicit_trinomial_probabilities",
    "log_explicit_price",
    "richardson_extrapolation",
    "PeacemanRachfordADI",
    "CraigSneydADI",
    "HestonADI",
    "heston_spot_grid",
    "heston_variance_grid",
]
