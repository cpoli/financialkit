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
assert day_count(last, settle, "30/360 US") == 122 and day_count(last, nxt, "30/360 US") == 180
assert np.isclose(accrued_interest(100.0, 0.10, last, settle, "30/360 US"), 5.0 * 122 / 180)  # Hull: 3.3889

# %%
# The end-of-month rules
# ----------------------

expected = {("2023-01-31", "2023-03-31"): (60, 60), ("2023-01-15", "2023-03-31"): (76, 75), ("2023-02-28", "2023-08-31"): (180, 182)}
for (start, end), days in expected.items():
    print(f"{start} -> {end}: 30/360 US {day_count(start, end, '30/360 US')}, 30E/360 {day_count(start, end, '30E/360')}")
    assert (day_count(start, end, "30/360 US"), day_count(start, end, "30E/360")) == days

# %%
# Accrued interest through a coupon period
# ----------------------------------------
# 30/360 accrues the same amount every month; actual days do not.

settles = np.arange(np.datetime64("2018-03-01"), np.datetime64("2018-09-01"))
fig, ax = plt.subplots(figsize=(8, 4.5))
ai_30360 = accrued_interest(100.0, 0.10, last, settles, "30/360 US")
ax.plot(settles, ai_30360, label="30/360 US")
firsts = settles.astype("datetime64[M]").astype("datetime64[D]") == settles
assert np.allclose(np.diff(ai_30360[firsts]), 10.0 / 12)
ax.plot(settles, accrued_interest(100.0, 0.10, last, settles, "ACT/365F"), label="ACT/365F")
ax.set_ylabel("accrued interest per 100")
ax.set_title("Accrued interest on a 10% semiannual bond")
ax.legend()
fig.autofmt_xdate()

plt.show()
