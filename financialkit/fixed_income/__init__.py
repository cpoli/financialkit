"""financialkit.fixed_income: bonds, duration, yield curves, immunization and swaps.

Bond cash flows, prices and yield to maturity (:func:`scipy.optimize.brentq`),
accrued interest and STRIPS; Macaulay, modified and Fisher-Weil duration,
convexity and Ho's key rate durations; Lutz forward rates, bootstrapping
a zero curve from bond prices or par rates (hand-written, solving each
node with :func:`~scipy.optimize.brentq`), Carleton-Cooper discount
factors and McCulloch's spline discount function
(:func:`numpy.linalg.lstsq`), Nelson-Siegel and Svensson fits
(:func:`scipy.optimize.least_squares`) and the principal components of
curve moves (:func:`numpy.linalg.eigh`); Redington immunization and the
Fong-Vasicek :math:`M^2` bound; interest-rate and currency swaps
valued as pairs of bonds, and overnight rates such as SOFR compounded in
arrears; OIS discounting with a separate forecasting curve; Treasury
bond futures (conversion factors and the cheapest to deliver); and
mortgage pass-through cash flows under the PSA prepayment benchmark.

Every curve is a :class:`YieldCurve` of continuously compounded zero
rates, and every function that takes a curve also accepts a flat rate.
Times are year fractions; converting dates under a day-count convention
belongs to :mod:`financialkit.time_value`.
"""

from financialkit import __version__
from financialkit.fixed_income.core.base import (
    BondRisk,
    CashFlows,
    CompoundedRate,
    CurveFitResult,
    CurveLike,
    CurvePCAResult,
    DeliveryBasis,
    FlatCurve,
    ImmunizationResult,
    KeyRateDurationResult,
    MBSCashFlows,
    MultiCurve,
    ShiftedCurve,
    SwapValuation,
    YieldCurve,
    ZeroCurve,
    as_curve,
    coupon_times,
)
from financialkit.fixed_income.systems.bonds import (
    accrued_interest,
    bond_accrued_interest,
    bond_cash_flows,
    bond_price,
    clean_price,
    price_from_yield,
    strips_curve,
    strips_decomposition,
    strips_value,
    yield_to_maturity,
    zero_coupon_bond_price,
)
from financialkit.fixed_income.systems.curves import (
    NelsonSiegelCurve,
    SplineDiscountCurve,
    SvenssonCurve,
    bootstrap_par_curve,
    bootstrap_zero_curve,
    carleton_cooper_discount_factors,
    fit_mcculloch_spline,
    fit_nelson_siegel,
    fit_svensson,
    forward_rates,
    yield_curve_pca,
)
from financialkit.fixed_income.systems.duration import (
    bond_risk,
    convexity,
    effective_duration,
    fisher_weil_convexity,
    fisher_weil_duration,
    key_rate_durations,
    key_rate_weights,
    macaulay_duration,
    modified_duration,
)
from financialkit.fixed_income.systems.futures import cheapest_to_deliver, conversion_factor, treasury_futures_price
from financialkit.fixed_income.systems.immunization import fong_vasicek_bound, horizon_value, linear_forward_shock, m_squared, redington_immunize
from financialkit.fixed_income.systems.mbs import mbs_cash_flows, psa_cpr, single_monthly_mortality
from financialkit.fixed_income.systems.multi_curve import bootstrap_forecast_curve, bootstrap_multi_curve
from financialkit.fixed_income.systems.swaps import (
    compounded_in_arrears,
    currency_swap_value,
    interest_rate_swap_value,
    overnight_forwards,
    swap_annuity,
    swap_rate,
)
from financialkit.fixed_income.utils.rates import yield_discount_derivatives
from financialkit.fixed_income.utils.synthetic import simulate_yield_changes
from financialkit.fixed_income.visualizers.plots import (
    plot_cash_flows,
    plot_discounted_cash_flows,
    plot_key_rate_durations,
    plot_mbs_cash_flows,
    plot_pca_loadings,
    plot_price_yield,
    plot_surplus,
    plot_yield_curve,
)

__all__ = [
    "__version__",
    "YieldCurve",
    "FlatCurve",
    "ZeroCurve",
    "ShiftedCurve",
    "CurveLike",
    "as_curve",
    "coupon_times",
    "CashFlows",
    "BondRisk",
    "CurveFitResult",
    "CurvePCAResult",
    "KeyRateDurationResult",
    "ImmunizationResult",
    "SwapValuation",
    "MultiCurve",
    "CompoundedRate",
    "DeliveryBasis",
    "MBSCashFlows",
    "bond_cash_flows",
    "price_from_yield",
    "bond_price",
    "yield_to_maturity",
    "bond_accrued_interest",
    "accrued_interest",
    "clean_price",
    "zero_coupon_bond_price",
    "strips_decomposition",
    "strips_value",
    "strips_curve",
    "bond_risk",
    "macaulay_duration",
    "modified_duration",
    "convexity",
    "fisher_weil_duration",
    "fisher_weil_convexity",
    "effective_duration",
    "key_rate_weights",
    "key_rate_durations",
    "forward_rates",
    "bootstrap_zero_curve",
    "bootstrap_par_curve",
    "carleton_cooper_discount_factors",
    "SplineDiscountCurve",
    "fit_mcculloch_spline",
    "NelsonSiegelCurve",
    "SvenssonCurve",
    "fit_nelson_siegel",
    "fit_svensson",
    "yield_curve_pca",
    "redington_immunize",
    "m_squared",
    "fong_vasicek_bound",
    "horizon_value",
    "linear_forward_shock",
    "swap_annuity",
    "swap_rate",
    "interest_rate_swap_value",
    "currency_swap_value",
    "compounded_in_arrears",
    "overnight_forwards",
    "bootstrap_forecast_curve",
    "bootstrap_multi_curve",
    "conversion_factor",
    "cheapest_to_deliver",
    "treasury_futures_price",
    "psa_cpr",
    "single_monthly_mortality",
    "mbs_cash_flows",
    "yield_discount_derivatives",
    "simulate_yield_changes",
    "plot_yield_curve",
    "plot_price_yield",
    "plot_discounted_cash_flows",
    "plot_cash_flows",
    "plot_key_rate_durations",
    "plot_pca_loadings",
    "plot_surplus",
    "plot_mbs_cash_flows",
]
