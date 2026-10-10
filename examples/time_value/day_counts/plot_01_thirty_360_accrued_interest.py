r"""
SIA 30/360 and accrued interest
===============================

A bond buyer pays the seller the coupon interest earned since the last
coupon date. The Securities Industry Association's *Standard Securities
Calculation Methods* (1993) codified how U.S. corporate and agency bonds
count those days: every month has 30 days and the year 360, with rules
for the 31st and for the end of February. The example is Hull's: a 10%
bond with coupons on March 1 and September 1, bought for settlement on
July 3, 2018.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_value import accrued_interest, day_count

# %%
# Hull's example
# --------------

last, settle, nxt = "2018-03-01", "2018-07-03", "2018-09-01"
print(f"30/360 days: {day_count(last, settle, '30/360 US')} of {day_count(last, nxt, '30/360 US')}")
print(f"actual days: {day_count(last, settle, 'ACT/360')} of {day_count(last, nxt, 'ACT/360')}")
print(f"accrued interest per 100: {accrued_interest(100.0, 0.10, last, settle, '30/360 US'):.4f}")

# %%
# The end-of-month rules
# ----------------------

for start, end in [("2023-01-31", "2023-03-31"), ("2023-01-15", "2023-03-31"), ("2023-02-28", "2023-08-31")]:
    print(f"{start} -> {end}: 30/360 US {day_count(start, end, '30/360 US')}, 30E/360 {day_count(start, end, '30E/360')}")

# %%
# Accrued interest through a coupon period
# ----------------------------------------
# 30/360 accrues the same amount every month; actual days do not.

settles = np.arange(np.datetime64("2018-03-01"), np.datetime64("2018-09-01"))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(settles, accrued_interest(100.0, 0.10, last, settles, "30/360 US"), label="30/360 US")
ax.plot(settles, accrued_interest(100.0, 0.10, last, settles, "ACT/365F"), label="ACT/365F")
ax.set_ylabel("accrued interest per 100")
ax.set_title("Accrued interest on a 10% semiannual bond")
ax.legend()
fig.autofmt_xdate()

plt.show()
