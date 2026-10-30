r"""
The PSA prepayment benchmark and MBS cash flows
===============================================

A mortgage pass-through passes the homeowners' payments on to investors,
and homeowners may prepay early. In 1985 the Public Securities
Association standardised the prepayment assumption. Under 100 PSA the
annual prepayment rate (CPR) rises by 0.2% a month to 6% at month 30 and
then stays there. Faster or slower speeds are multiples of that ramp.
The speed sets the timing of the principal, and so the security's
average life and price.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import mbs_cash_flows, plot_mbs_cash_flows, price_from_yield, psa_cpr

# %%
# The PSA ramp
# ------------

age = np.arange(1, 121)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for psa in (50, 100, 200, 300):
    axes[0].plot(age, 100 * psa_cpr(age, psa), label=f"{psa} PSA")
axes[0].set_xlabel("loan age (months)")
axes[0].set_ylabel("CPR (%)")
axes[0].legend()

# %%
# Fabozzi's pass-through
# ----------------------
#
# A $400 million pool with an 8.125% weighted average coupon, a 7.5%
# pass-through rate and 357 months left, at 100 PSA.

pool = mbs_cash_flows(400e6, 0.08125, 357, 0.075, psa=100, age=3)
print(f"month 1: interest {pool.interest[0]:,.0f}, scheduled principal {pool.scheduled_principal[0]:,.0f}, prepayment {pool.prepayment[0]:,.0f}")
assert np.allclose([pool.interest[0], pool.scheduled_principal[0], pool.prepayment[0]], [2_500_000, 267_535, 267_470], atol=0.5)
assert abs(pool.principal.sum() - 400e6) < 1e-3
plot_mbs_cash_flows(mbs_cash_flows(400e6, 0.08125, 357, 0.075, psa=300, age=3), ax=axes[1], label="300 PSA")

# %%
# Speed, average life and price
# -----------------------------
#
# Faster prepayment returns principal sooner. For a premium pool, priced
# above par, that hurts, because the principal comes back at par. For a
# discount pool it helps.

speeds = np.arange(0, 601, 25)
pools = [mbs_cash_flows(100.0, 0.07, 360, 0.065, psa=s) for s in speeds]
wal = np.array([p.weighted_average_life for p in pools])
premium = np.array([price_from_yield(p.as_cash_flows(), 0.05, compounding=12) for p in pools])
discount = np.array([price_from_yield(p.as_cash_flows(), 0.08, compounding=12) for p in pools])
print(f"WAL at 0, 100 and 300 PSA: {wal[0]:.1f}, {wal[4]:.1f}, {wal[12]:.1f} years")
assert np.all(np.diff(wal) < 0) and np.all(np.diff(premium) < 0) and np.all(np.diff(discount) > 0)

ax = axes[2]
ax.plot(speeds, premium, label="priced at a 5% yield")
ax.plot(speeds, discount, label="priced at an 8% yield")
ax.axhline(100.0, color="0.6", lw=0.8)
ax.set_xlabel("prepayment speed (PSA)")
ax.set_ylabel("price of a 6.5% pass-through")
ax.legend()
twin = ax.twinx()
twin.plot(speeds, wal, "k--", lw=1)
twin.set_ylabel("weighted average life (years, dashed)")
plt.tight_layout()
plt.show()
