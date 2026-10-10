"""Supporting numerics for financialkit.fixed_income's systems/ modules."""

from financialkit.fixed_income.utils.rates import yield_discount_derivatives
from financialkit.fixed_income.utils.synthetic import simulate_yield_changes

__all__ = [
    "yield_discount_derivatives",
    "simulate_yield_changes",
]
