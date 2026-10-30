r"""
Lorie and Savage: multiple IRRs
===============================

In "Three Problems in Rationing Capital" (*Journal of Business*, 1955)
James Lorie and Leonard Savage pointed out that the IRR can fail. Their
example: replacing an oil-well pump with a bigger one costs 1,600 now,
brings in 10,000 extra next year, and loses the 10,000 the old pump
would have produced the year after. The NPV is zero at both 25% and
400%, and negative outside them. By Descartes' rule of signs a stream
with :math:`k` sign changes can have up to :math:`k` IRRs;
:func:`~financialkit.time_value.irr_all` finds every one as a root of
the NPV polynomial in :math:`1/(1 + r)`, and
:func:`~financialkit.time_value.irr` warns when there is more than one.
"""

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_value import MultipleIRRWarning, irr, irr_all, npv_profile, plot_npv_profile, sign_changes

# %%
# The pump
# --------

pump = [-1600.0, 10_000.0, -10_000.0]
print(f"sign changes: {sign_changes(pump)}")
print(f"all IRRs: {irr_all(pump)}")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    irr(pump)
print("irr() warned:", any(issubclass(w.category, MultipleIRRWarning) for w in caught))
assert sign_changes(pump) == 2 and np.allclose(irr_all(pump), [0.25, 4.0])
assert any(issubclass(w.category, MultipleIRRWarning) for w in caught)

# %%
# The NPV profile is a hump
# -------------------------
# Is a 25% IRR "good"? Is 400%? Neither tells the investor anything; the
# NPV at the cost of capital does.

fig, ax = plt.subplots(figsize=(7, 4.5))
plot_npv_profile(npv_profile(pump, np.linspace(0.0, 5.0, 501)), ax=ax, label="pump project")
ax.legend()
ax.set_title("Lorie-Savage pump: two IRRs")

plt.show()
