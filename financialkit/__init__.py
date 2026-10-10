"""financialkit: unified toolkit for quantitative finance.

Import as ``fk`` by convention::

    import financialkit as fk
    times, paths = fk.integrators.gbm_paths(100.0, 0.05, 0.2, T=1.0, n_steps=252, n_paths=1000, rng=0)

Educational and research software, not investment advice.

Domains are added incrementally; see ``__all__`` for what's currently
implemented, and ROADMAP.md for the full plan.
"""

# Defined before the subpackage imports: each subpackage re-exports it via
# ``from financialkit import __version__`` while this module is still loading.
__version__ = "0.1.0"

from financialkit import (  # noqa: E402
    asset_pricing,
    constants,
    credit,
    exotic_options,
    fixed_income,
    integrators,
    interest_rate_models,
    market_microstructure,
    monte_carlo,
    options,
    pde_methods,
    portfolio,
    risk,
    stochastic_volatility,
    time_series,
    time_value,
)

__all__ = [
    "asset_pricing",
    "constants",
    "credit",
    "exotic_options",
    "fixed_income",
    "integrators",
    "interest_rate_models",
    "market_microstructure",
    "monte_carlo",
    "options",
    "pde_methods",
    "portfolio",
    "risk",
    "stochastic_volatility",
    "time_series",
    "time_value",
]
