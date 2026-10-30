r"""
Modigliani-Miller
=================

Franco Modigliani and Merton Miller (1958) showed that, without taxes
or frictions, a firm's value does not depend on how it is financed
(Proposition I). Debt is cheaper than equity, but borrowing makes the
remaining equity riskier, and its required return rises exactly enough
to keep the weighted average cost of capital constant (Proposition II):

.. math::

   r_E = r_A + (r_A - r_D)\,\frac{D}{E}.

In 1963 they added corporate tax: interest is deductible, so each unit
of permanent debt adds a tax shield worth :math:`t D`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing import levered_cost_of_equity, levered_firm_value, wacc

rA, rD = 0.10, 0.05
L = np.linspace(0.0, 0.8, 100)  # debt / value
de = L / (1 - L)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for t, ls in [(0.0, "-"), (0.3, "--")]:
    rE = levered_cost_of_equity(rA, rD, de, t)
    if t == 0.0:
        # Without tax the cost of equity rises exactly enough to hold the WACC at r_A.
        assert np.allclose(wacc(rE, rD, L, t), rA)
    axes[0].plot(L, rE, "C3", ls=ls, label=f"r_E, t = {t:.0%}")
    axes[0].plot(L, wacc(rE, rD, L, t), "C0", ls=ls, label=f"WACC, t = {t:.0%}")
axes[0].axhline(rD, color="0.5", lw=1, label="r_D")
axes[0].set_xlabel("D / V")
axes[0].set_ylabel("required return")
axes[0].set_title("Proposition II")
axes[0].legend(fontsize=8)

# %%
# Proposition I
# -------------

VU = 100.0
D = np.linspace(0, 80, 50)
for t in (0.0, 0.2, 0.35):
    assert np.allclose(levered_firm_value(VU, D, t), VU + t * D)
    axes[1].plot(D, levered_firm_value(VU, D, t), label=f"t = {t:.0%}")
axes[1].set_xlabel("debt D")
axes[1].set_ylabel("firm value V_L")
axes[1].set_title("Proposition I: V_L = V_U + t D")
axes[1].legend()
fig.tight_layout()

plt.show()
