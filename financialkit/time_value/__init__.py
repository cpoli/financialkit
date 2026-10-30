"""financialkit.time_value: the time value of money.

Compounding at any frequency and the equivalence of rates (continuous by
default); Pacioli's rule of 72 and Witt's compound-interest tables;
present value, Fisher's net present value and real rate of interest;
annuities, perpetuities, De Witt's life annuities and Price's sinking
fund; the internal rate of return (:func:`scipy.optimize.brentq`), every
root of the NPV polynomial (:func:`numpy.roots`) with a warning when
there are several, and Baldwin's modified IRR; level-payment
amortization schedules and the Truth-in-Lending APR; day counts, year
fractions and accrued interest for every
:class:`~financialkit.constants.DayCount` convention; coupon schedules
with short or long stubs and the end-of-month rule, business-day
adjustments on a weekend calendar plus user-supplied holidays
(:func:`numpy.busday_offset`) and settlement lags.

The year-fraction functions here are the package's single
implementation; later domains import them rather than redefining them.
"""

from financialkit import __version__
from financialkit.time_value.core.base import AmortizationSchedule, CompoundingTable, MultipleIRRWarning, NPVProfile, Schedule, SinkingFundSchedule
from financialkit.time_value.systems.amortization import amortization_schedule, annual_percentage_rate, loan_payment
from financialkit.time_value.systems.annuities import (
    annuity_factor,
    annuity_future_value,
    annuity_payment,
    annuity_present_value,
    de_witt_survival,
    life_annuity_value,
    perpetuity_present_value,
    sinking_fund_payment,
    sinking_fund_schedule,
)
from financialkit.time_value.systems.compounding import (
    compound_interest_table,
    continuous_rate,
    discount_factor,
    doubling_time,
    effective_annual_rate,
    equivalent_rate,
    future_value,
    rule_of_72,
)
from financialkit.time_value.systems.day_counts import accrued_interest, day_count, year_fraction
from financialkit.time_value.systems.irr import irr, irr_all, mirr, npv_profile
from financialkit.time_value.systems.present_value import fisher_nominal_rate, fisher_real_rate, npv, present_value
from financialkit.time_value.systems.schedules import BUSINESS_DAY_CONVENTIONS, STUBS, adjust_date, coupon_schedule, is_business_day, settlement_date
from financialkit.time_value.utils.cash_flows import sign_changes
from financialkit.time_value.utils.dates import add_months, days_in_year, is_end_of_month, is_leap_year
from financialkit.time_value.visualizers.plots import plot_amortization, plot_cash_flows, plot_compounding_growth, plot_npv_profile

__all__ = [
    "__version__",
    "CompoundingTable",
    "NPVProfile",
    "AmortizationSchedule",
    "SinkingFundSchedule",
    "Schedule",
    "MultipleIRRWarning",
    "future_value",
    "discount_factor",
    "continuous_rate",
    "equivalent_rate",
    "effective_annual_rate",
    "doubling_time",
    "rule_of_72",
    "compound_interest_table",
    "present_value",
    "npv",
    "fisher_real_rate",
    "fisher_nominal_rate",
    "annuity_factor",
    "annuity_present_value",
    "annuity_future_value",
    "annuity_payment",
    "perpetuity_present_value",
    "de_witt_survival",
    "life_annuity_value",
    "sinking_fund_payment",
    "sinking_fund_schedule",
    "irr",
    "irr_all",
    "mirr",
    "npv_profile",
    "loan_payment",
    "amortization_schedule",
    "annual_percentage_rate",
    "day_count",
    "year_fraction",
    "accrued_interest",
    "BUSINESS_DAY_CONVENTIONS",
    "STUBS",
    "is_business_day",
    "adjust_date",
    "settlement_date",
    "coupon_schedule",
    "sign_changes",
    "is_leap_year",
    "days_in_year",
    "is_end_of_month",
    "add_months",
    "plot_cash_flows",
    "plot_compounding_growth",
    "plot_npv_profile",
    "plot_amortization",
]
