"""Credit models: scoring, structural and first-passage models, hazard rates
and recovery, CDS pricing and bootstrapping, rating migration, and
portfolio losses (Vasicek, CreditRisk+, copulas, Basel IRB, tranches).
"""

from financialkit.credit.systems.basel import basel_correlation, basel_irb_capital, basel_maturity_adjustment, basel_risk_weight
from financialkit.credit.systems.cds import bootstrap_hazard_curve, cds_legs, cds_par_spread, cds_upfront, upfront_to_spread
from financialkit.credit.systems.copula_defaults import copula_default_times, default_correlation, joint_default_probability, nth_to_default_probability
from financialkit.credit.systems.creditrisk_plus import creditrisk_plus
from financialkit.credit.systems.first_passage import black_cox_default_probability, black_cox_survival, first_passage_crossing_probability
from financialkit.credit.systems.hazard import HazardCurve, hazard_from_spread, jarrow_turnbull_zero, spread_from_hazard
from financialkit.credit.systems.leland import leland_model, leland_optimal_coupon
from financialkit.credit.systems.migration import (
    creditmetrics_bond,
    creditmetrics_portfolio,
    generator_from_transition,
    is_valid_generator,
    jlt_risky_zero,
    joint_migration_probabilities,
    migration_thresholds,
    regularize_generator,
    simulate_migrations,
    transition_from_generator,
)
from financialkit.credit.systems.recovery import duffie_singleton_cir_zero, risky_zero_coupon
from financialkit.credit.systems.scoring import ALTMAN_COEFFICIENTS, altman_ratios, altman_z_score, altman_zone, linear_discriminant
from financialkit.credit.systems.structural import distance_to_default, kmv_default_point, kmv_iterate, merton_calibrate, merton_model
from financialkit.credit.systems.tranches import expected_tranche_loss, tranche_loss, tranche_spread
from financialkit.credit.systems.vasicek import (
    conditional_default_probability,
    one_factor_loss_distribution,
    simulate_one_factor_defaults,
    vasicek_cdf,
    vasicek_pdf,
    vasicek_quantile,
)

__all__ = [
    "ALTMAN_COEFFICIENTS",
    "altman_ratios",
    "altman_z_score",
    "altman_zone",
    "linear_discriminant",
    "merton_model",
    "merton_calibrate",
    "distance_to_default",
    "kmv_default_point",
    "kmv_iterate",
    "black_cox_survival",
    "black_cox_default_probability",
    "first_passage_crossing_probability",
    "leland_model",
    "leland_optimal_coupon",
    "HazardCurve",
    "jarrow_turnbull_zero",
    "hazard_from_spread",
    "spread_from_hazard",
    "risky_zero_coupon",
    "duffie_singleton_cir_zero",
    "cds_legs",
    "cds_par_spread",
    "cds_upfront",
    "upfront_to_spread",
    "bootstrap_hazard_curve",
    "migration_thresholds",
    "creditmetrics_bond",
    "joint_migration_probabilities",
    "simulate_migrations",
    "creditmetrics_portfolio",
    "generator_from_transition",
    "transition_from_generator",
    "is_valid_generator",
    "regularize_generator",
    "jlt_risky_zero",
    "conditional_default_probability",
    "vasicek_cdf",
    "vasicek_pdf",
    "vasicek_quantile",
    "simulate_one_factor_defaults",
    "one_factor_loss_distribution",
    "creditrisk_plus",
    "copula_default_times",
    "joint_default_probability",
    "default_correlation",
    "nth_to_default_probability",
    "basel_correlation",
    "basel_maturity_adjustment",
    "basel_irb_capital",
    "basel_risk_weight",
    "tranche_loss",
    "expected_tranche_loss",
    "tranche_spread",
]
