"""financialkit.pde_methods: finite-difference methods for option pricing.

The Black-Scholes equation on a spot grid by the theta-method: Schwartz's
explicit scheme with its stability check, Laasonen's implicit scheme and
Crank-Nicolson with Rannacher's implicit start, the tridiagonal systems
solved by :func:`scipy.linalg.solve_banded` or a hand-written Thomas
algorithm, uniform and Tavella-Randall grids, and knock-out barriers on a
barrier-aligned grid. American options by Cryer's projected SOR, the
Brennan-Schwartz direct solve and Forsyth-Vetzal's penalty iteration.
The theory around the schemes: von Neumann amplification factors, the
explicit log-spot scheme as a trinomial tree, the transformation to the
heat equation and Richardson extrapolation. Two assets by the
Peaceman-Rachford and Craig-Sneyd ADI schemes, and the Heston PDE by
in 't Hout and Foulon's ADI schemes on non-uniform grids.
"""

from financialkit import __version__
from financialkit.pde_methods.core.base import (
    ADIResult,
    FDResult,
    FiniteDifferencePricer,
    HeatEquationResult,
    HestonPDEResult,
    LCPSolution,
    RichardsonResult,
    StabilityResult,
    TrinomialProbabilities,
    VonNeumannResult,
)
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
from financialkit.pde_methods.utils.operators import black_scholes_operator, discounted_intrinsic, interpolate_at, solve_tridiagonal, tridiagonal_matrix
from financialkit.pde_methods.utils.splitting import ADI_SCHEMES, adi_march, default_adi_theta, difference_matrices
from financialkit.pde_methods.utils.stepping import theta_march
from financialkit.pde_methods.visualizers.plots import (
    plot_adi_surface,
    plot_amplification,
    plot_exercise_boundary,
    plot_grid,
    plot_grid_convergence,
    plot_heat_solution,
    plot_heston_surface,
    plot_value_profile,
    plot_value_surface,
)

__all__ = [
    "__version__",
    "FDResult",
    "HeatEquationResult",
    "TrinomialProbabilities",
    "ADIResult",
    "HestonPDEResult",
    "LCPSolution",
    "StabilityResult",
    "VonNeumannResult",
    "RichardsonResult",
    "FiniteDifferencePricer",
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
    "black_scholes_operator",
    "tridiagonal_matrix",
    "solve_tridiagonal",
    "discounted_intrinsic",
    "interpolate_at",
    "theta_march",
    "ADI_SCHEMES",
    "adi_march",
    "default_adi_theta",
    "difference_matrices",
    "plot_value_profile",
    "plot_value_surface",
    "plot_exercise_boundary",
    "plot_amplification",
    "plot_grid_convergence",
    "plot_grid",
    "plot_heat_solution",
    "plot_adi_surface",
    "plot_heston_surface",
]
