"""Supporting numerics and data for financialkit.credit's systems/ modules."""

from financialkit.credit.utils.discounting import discount_factors, zero_rate_to
from financialkit.credit.utils.matrices import RATINGS, check_transition_matrix, creditmetrics_bbb_values, creditmetrics_matrix
from financialkit.credit.utils.numerics import loss_recursion, normal_factor_quadrature, panjer_recursion

__all__ = [
    "RATINGS",
    "creditmetrics_matrix",
    "creditmetrics_bbb_values",
    "check_transition_matrix",
    "panjer_recursion",
    "loss_recursion",
    "normal_factor_quadrature",
    "discount_factors",
    "zero_rate_to",
]
