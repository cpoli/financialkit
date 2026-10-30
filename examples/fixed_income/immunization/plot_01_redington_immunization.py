r"""
Redington immunization
======================

Frank Redington (1952), an actuary, asked how a life office could hold
assets so that a change in interest rates leaves its surplus unharmed.
His answer has three conditions: match the present values, match the
durations, and make the assets more convex than the liabilities. Then the
surplus has a local minimum at today's rates. Assets that straddle the
liabilities satisfy the third condition. Assets bunched inside them do
not.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import CashFlows, bond_cash_flows, redington_immunize
from financialkit.fixed_income.visualizers import plot_surplus

curve = 0.04
liabilities = CashFlows([3.0, 6.0, 9.0, 12.0], [250.0, 250.0, 250.0, 250.0])

straddle = redington_immunize(liabilities, [bond_cash_flows(1.0, 0.0, 1.0), bond_cash_flows(1.0, 0.0, 20.0)], curve)
inside = redington_immunize(liabilities, [bond_cash_flows(1.0, 0.0, 6.5), bond_cash_flows(1.0, 0.0, 8.0)], curve)

for name, res in (("1y/20y zeros", straddle), ("6.5y/8y zeros", inside)):
    print(
        f"{name}: holdings {res.holdings.round(1)}, PV {res.pv_assets:.2f} vs {res.pv_liabilities:.2f}, "
        f"D {res.duration_assets:.3f} vs {res.duration_liabilities:.3f}, "
        f"C {res.convexity_assets:.1f} vs {res.convexity_liabilities:.1f}, Redington: {res.redington_satisfied}"
    )
assert straddle.redington_satisfied and not inside.redington_satisfied

# %%
# Surplus under parallel shifts
# -----------------------------

fig, ax = plt.subplots()
shifts = np.linspace(-0.03, 0.03, 61)
assert np.all(straddle.surplus(shifts) > -1e-9)  # a minimum at today's rates
assert np.all(inside.surplus(shifts) < 1e-9) and inside.surplus(shifts).min() < -1.0  # a maximum
plot_surplus(straddle, shifts, ax=ax)
plot_surplus(inside, shifts, ax=ax)
ax.legend(["assets straddle liabilities", "_", "assets inside liabilities"])
ax.set_title("Surplus = assets - liabilities")
plt.show()
