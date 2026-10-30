r"""
SOFR: compounding in arrears
============================

The New York Fed first published the Secured Overnight Financing Rate on
3 April 2018, as the replacement for US dollar LIBOR. A LIBOR coupon was
fixed at the start of its period. A SOFR coupon compounds each business
day's overnight rate over the period, so it is known only at the end.
The ARRC conventions (a lookback, an observation shift, a lockout) move
the observation window a few days earlier, so the payment amount is
known in time to pay it.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import NelsonSiegelCurve, compounded_in_arrears, overnight_forwards

# %%
# A synthetic overnight rate
# --------------------------
#
# The rate steps up 25 bp after two policy meetings, adds small daily
# noise and spikes at each month end.

rng = np.random.default_rng(7)
days = np.arange("2024-12-02", "2025-07-01", dtype="datetime64[D]")
holidays = np.array(["2024-12-25", "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19"], dtype="datetime64[D]")
days = days[np.is_busday(days, holidays=holidays)]
sofr = 0.0430 + 0.0025 * (days >= np.datetime64("2025-03-20")) + 0.0025 * (days >= np.datetime64("2025-05-08"))
month_end = days[1:].astype("datetime64[M]") != days[:-1].astype("datetime64[M]")
sofr = sofr + 0.0003 * rng.standard_normal(days.size) + 0.0010 * np.append(month_end, False)

start, end = "2025-03-03", "2025-06-02"
plain = compounded_in_arrears(days, sofr, start, end, holidays=holidays)
print(f"compounded SOFR {100 * plain.rate:.4f}% over {plain.weights.sum()} days; coupon on $100m {1e8 * plain.rate * plain.accrual:,.0f}")
assert plain.weights.sum() == 91 and plain.fixing_dates[-1] == np.datetime64("2025-05-30")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].step(days, 100 * sofr, where="post", lw=0.8, label="daily SOFR")
axes[0].plot(plain.dates, 100 * plain.cumulative, lw=2, label="compounded so far")
axes[0].axvspan(np.datetime64(start), np.datetime64(end), color="0.9")
axes[0].set_ylabel("rate (%)")
axes[0].legend()
fig.autofmt_xdate()

# %%
# The ARRC conventions
# --------------------
#
# Each convention changes the coupon by a few basis points at most. What
# it changes is when the coupon is known: 2 business days before payment
# for a 2-day lookback or a 2-day lockout.

conventions = {
    "plain": plain,
    "2-day lookback": compounded_in_arrears(days, sofr, start, end, lookback=2, holidays=holidays),
    "2-day observation shift": compounded_in_arrears(days, sofr, start, end, lookback=2, observation_shift=True, holidays=holidays),
    "2-day lockout": compounded_in_arrears(days, sofr, start, end, lockout=2, holidays=holidays),
}
for name, r in conventions.items():
    print(f"{name:24s} {100 * r.rate:.4f}%   last fixing used {r.fixing_dates[-1]}")
assert max(abs(r.rate - plain.rate) for r in conventions.values()) < 5e-4
axes[1].barh(list(conventions), [1e4 * (r.rate - plain.rate) for r in conventions.values()])
axes[1].set_xlabel("difference from plain compounding (bp)")

# %%
# The expected coupon is the OIS forward
# --------------------------------------
#
# Compounding the overnight forwards a curve implies telescopes to
# :math:`P(t_0)/P(t_n)`. The expected coupon is therefore the curve's
# simple forward rate, and a SOFR leg is worth :math:`P(t_0) - P(t_n)`.

curve = NelsonSiegelCurve(0.040, 0.004, -0.005, 1.5)
f_dates, f_rates = overnight_forwards(curve, start, end, valuation_date="2024-12-02", holidays=holidays)
expected = compounded_in_arrears(f_dates, f_rates, start, end, holidays=holidays)
P = curve.discount(np.array([91, 182]) / 365)
assert abs(expected.rate - (P[0] / P[1] - 1) / expected.accrual) < 1e-13
print(f"expected coupon rate {100 * expected.rate:.4f}% = simple forward {100 * (P[0] / P[1] - 1) / expected.accrual:.4f}%")
plt.tight_layout()
plt.show()
