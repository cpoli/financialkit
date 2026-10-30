r"""
Pacioli's rule of 72
====================

Luca Pacioli's *Summa de arithmetica* (1494) gives, without proof, a rule
for how long money takes to double: divide 72 by the annual interest
rate in percent. The exact doubling time with annual compounding is
:math:`\ln 2 / \ln(1 + r)`, and since :math:`\ln(1 + r) \approx r - r^2/2`
the best constant drifts upward with the rate; 72 is close to right near
8% and has many convenient divisors.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_value import doubling_time, rule_of_72

# %%
# The rule against the exact answer
# ---------------------------------

for r in (0.02, 0.06, 0.08, 0.12, 0.24):
    print(f"{r:5.0%}: rule {rule_of_72(r):6.2f} years, exact {doubling_time(r, compounding=1):6.2f} years")

# %%
# Relative error of the rules of 69.3, 70 and 72
# -----------------------------------------------
# 69.3 is exact for continuous compounding; with annual compounding a
# larger constant is better.

r = np.linspace(0.01, 0.30, 300)
exact = doubling_time(r, compounding=1)
fig, ax = plt.subplots(figsize=(7, 4.5))
at_8 = {}
for c in (69.3, 70.0, 72.0):
    ax.plot(100 * r, 100 * (rule_of_72(r, numerator=c) / exact - 1), label=f"rule of {c:g}")
    at_8[c] = abs(rule_of_72(0.08, numerator=c) / doubling_time(0.08, compounding=1) - 1)
assert min(at_8, key=at_8.get) == 72.0 and at_8[72.0] < 1e-3
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xlabel("annual rate (%)")
ax.set_ylabel("error vs exact doubling time (%)")
ax.set_title("Pacioli's rule of 72")
ax.legend()

plt.show()
