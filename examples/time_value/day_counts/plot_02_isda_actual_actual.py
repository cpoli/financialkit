r"""
ISDA's actual/actual and the day-count zoo
==========================================

When the euro replaced its legacy currencies, markets had to agree on
how to count days. ISDA's 1998 memo *EMU and Market Conventions* set
the competing "actual/actual" methods side by side; under the ISDA
variant, the days falling in each calendar year are divided by that
year's length, 365 or 366. Its example: November 1, 2003 to May 1, 2004
is 61/365 + 121/366 = 0.49772. This script compares that with the other
conventions in :class:`~financialkit.constants.DayCount`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.constants import DayCount
from financialkit.time_value import year_fraction

# %%
# ISDA's example under every convention
# -------------------------------------

for dc in DayCount:
    print(f"{dc.value:13s} {year_fraction('2003-11-01', '2004-05-01', dc):.5f}")

# %%
# Six-month periods through a leap year
# -------------------------------------
# The same calendar period earns different interest under each convention.

starts = np.arange(np.datetime64("2023-01-01"), np.datetime64("2025-01-01"), 7)
ends = (starts.astype("datetime64[M]") + 6).astype("datetime64[D]") + (starts - starts.astype("datetime64[M]").astype("datetime64[D]"))
fig, ax = plt.subplots(figsize=(8, 4.5))
for dc in DayCount:
    ax.plot(starts, year_fraction(starts, ends, dc), label=dc.value)
ax.axhline(0.5, color="k", lw=0.8, ls=":")
ax.set_ylabel("year fraction of a six-month period")
ax.set_title("The day-count zoo")
ax.legend(fontsize=8)
fig.autofmt_xdate()

plt.show()
