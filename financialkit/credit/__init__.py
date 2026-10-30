"""financialkit.credit: credit scoring, structural and reduced-form default models, CDS, rating migration and portfolio credit risk.

Altman's Z-score; Merton's structural model, its calibration to equity
and Crosbie-Bohn distance to default (through
:func:`financialkit.options.black_scholes_price`); Black-Cox first
passage and Leland's endogenous default; Jarrow-Turnbull hazard curves
and Duffie-Singleton recovery of market value; O'Kane-Turnbull CDS legs,
par spreads, upfronts and the ISDA-style hazard bootstrap; CreditMetrics
migration, Jarrow-Lando-Turnbull generators (:func:`scipy.linalg.expm`)
and Israel-Rosenthal-Wei regularization; Vasicek's large homogeneous
portfolio, CreditRisk+ by Panjer's recursion, Li's Gaussian copula, the
Basel II IRB formula and Hull-White tranche pricing; the
Andersen-Sidenius-Basu loss recursion and base correlation; and
counterparty exposure by regression on Hull-White paths
(:class:`financialkit.interest_rate_models.HullWhiteModel`) with
Brigo-Masetti CVA, Gregory's bilateral CVA and DVA and Burgard-Kjaer
FVA. Time is in years,
rates are continuously compounded and ``recovery`` is a fraction of face
value.
"""

from financialkit import __version__
from financialkit.credit.core.base import (
    BaseCorrelationResult,
    BilateralCVAResult,
    CDSResult,
    CreditMetricsResult,
    ExposureProfile,
    FVAResult,
    KMVResult,
    LelandResult,
    LossDistribution,
    MertonResult,
    SurvivalCurve,
    TrancheResult,
)
from financialkit.credit.systems.basel import basel_correlation, basel_irb_capital, basel_maturity_adjustment, basel_risk_weight
from financialkit.credit.systems.cds import bootstrap_hazard_curve, cds_legs, cds_par_spread, cds_upfront, upfront_to_spread
from financialkit.credit.systems.copula_defaults import copula_default_times, default_correlation, joint_default_probability, nth_to_default_probability
from financialkit.credit.systems.creditrisk_plus import creditrisk_plus
from financialkit.credit.systems.exposure import pathwise_discount, regression_mark_to_market, swap_exposure
from financialkit.credit.systems.first_passage import black_cox_default_probability, black_cox_survival, first_passage_crossing_probability
from financialkit.credit.systems.hazard import HazardCurve, hazard_from_spread, jarrow_turnbull_zero, spread_from_hazard
from financialkit.credit.systems.leland import leland_model, leland_optimal_coupon
from financialkit.credit.systems.loss_recursion import asb_loss_distribution
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
from financialkit.credit.systems.tranches import (
    base_correlation,
    base_correlation_spread,
    compound_correlations,
    expected_tranche_loss,
    tranche_loss,
    tranche_spread,
)
from financialkit.credit.systems.vasicek import (
    conditional_default_probability,
    one_factor_loss_distribution,
    simulate_one_factor_defaults,
    vasicek_cdf,
    vasicek_pdf,
    vasicek_quantile,
)
from financialkit.credit.systems.xva import bilateral_cva, funding_valuation_adjustment, swap_cva_swaptions, unilateral_cva
from financialkit.credit.utils.matrices import RATINGS, check_transition_matrix, creditmetrics_bbb_values, creditmetrics_matrix
from financialkit.credit.utils.numerics import loss_recursion, normal_factor_quadrature, panjer_recursion
from financialkit.credit.visualizers.plots import (
    plot_exposure_profile,
    plot_loss_distribution,
    plot_survival,
    plot_tranche_losses,
    plot_transition_matrix,
    plot_value_distribution,
)

__all__ = [
    "__version__",
    "SurvivalCurve",
    "MertonResult",
    "KMVResult",
    "LelandResult",
    "CDSResult",
    "CreditMetricsResult",
    "LossDistribution",
    "TrancheResult",
    "BaseCorrelationResult",
    "ExposureProfile",
    "BilateralCVAResult",
    "FVAResult",
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
    "asb_loss_distribution",
    "base_correlation_spread",
    "base_correlation",
    "compound_correlations",
    "regression_mark_to_market",
    "pathwise_discount",
    "swap_exposure",
    "unilateral_cva",
    "swap_cva_swaptions",
    "bilateral_cva",
    "funding_valuation_adjustment",
    "RATINGS",
    "creditmetrics_matrix",
    "creditmetrics_bbb_values",
    "check_transition_matrix",
    "panjer_recursion",
    "loss_recursion",
    "normal_factor_quadrature",
    "plot_survival",
    "plot_loss_distribution",
    "plot_transition_matrix",
    "plot_value_distribution",
    "plot_tranche_losses",
    "plot_exposure_profile",
]
