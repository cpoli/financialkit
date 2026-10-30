r"""
Fisher-Weil duration
====================

Fisher and Weil (1971) computed Macaulay's weights with each cash flow
discounted on its own zero rate instead of the bond's single yield,

.. math:: D_{FW} = \frac{1}{P}\sum_i t_i\, c_i\, e^{-r(t_i) t_i}.

It is the exact sensitivity of the price to a parallel shift of the zero
curve, which the script verifies by repricing on shifted curves.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import NelsonSiegelCurve, bond_cash_flows, fisher_weil_duration, macaulay_duration, yield_to_maturity

curve = NelsonSiegelCurve(0.05, -0.035, 0.0, 3.0)  # steeply upward sloping
cf = bond_cash_flows(100.0, 0.06, 20.0)
P = cf.present_value(curve)
ytm = yield_to_maturity(P, cf)
d_fw = fisher_weil_duration(cf, curve)
print(f"price {P:.3f}, yield {100 * ytm:.3f}%")
print(f"Macaulay at the yield: {macaulay_duration(cf, ytm):.4f}")
print(f"Fisher-Weil on the curve: {d_fw:.4f}")

# %%
# Parallel shifts of the zero curve
# ---------------------------------

shifts = np.linspace(-0.02, 0.02, 41)
actual = np.array([cf.present_value(curve.shifted(s)) for s in shifts])
fig, ax = plt.subplots()
ax.plot(1e4 * shifts, actual, label="repriced")
ax.plot(1e4 * shifts, P * np.exp(-d_fw * shifts), ls="--", label=r"$P e^{-D_{FW} s}$")
ax.set_xlabel("parallel shift of zero curve (bp)")
ax.set_ylabel("price")
ax.legend()
slope = (actual[21] - actual[19]) / (shifts[21] - shifts[19]) / P
print(f"-dP/ds / P by finite difference: {-slope:.4f}")
assert abs(-slope - d_fw) < 1e-4 * d_fw  # even with 10 bp steps
plt.show()
