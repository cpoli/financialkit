"""Concrete time-value-of-money models: compounding, present value,
annuities, IRR, amortization and day counts.
"""

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

__all__ = [
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
]
