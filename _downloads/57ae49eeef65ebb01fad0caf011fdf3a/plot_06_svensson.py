r"""
Svensson
========

Lars Svensson (1994) added a second hump term to Nelson-Siegel, with its
own decay :math:`\tau_2`, to fit forward curves that bend twice. He built
it to estimate forward rates at the Swedish Riksbank, and central banks
still publish curves fitted with it. On such a curve, Nelson-Siegel
misses the second bend and Svensson does not.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import SvenssonCurve, fit_nelson_siegel, fit_svensson

tenors = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30])
true = SvenssonCurve(0.045, -0.02, -0.03, 0.045, 0.8, 7.0)
rng = np.random.default_rng(1994)
observed = true.zero_rate(tenors) + rng.normal(0, 0.0002, tenors.size)

ns = fit_nelson_siegel(tenors, observed)
sv = fit_svensson(tenors, observed)
print(f"Nelson-Siegel RMSE {1e4 * ns.rmse:.2f} bp, Svensson RMSE {1e4 * sv.rmse:.2f} bp")

# %%
# Zero and forward curves
# -----------------------

t = np.linspace(0.05, 30, 300)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(tenors, 100 * observed, "ko", label="observed")
for name, res in (("Nelson-Siegel", ns), ("Svensson", sv)):
    axes[0].plot(t, 100 * res.curve.zero_rate(t), label=name)
    axes[1].plot(t, 100 * res.curve.instantaneous_forward(t), label=name)
axes[1].plot(t, 100 * true.instantaneous_forward(t), "k:", label="true")
axes[0].set_title("zero rates")
axes[1].set_title("instantaneous forwards")
for ax in axes:
    ax.set_xlabel("maturity (years)")
    ax.set_ylabel("rate (%)")
    ax.legend()
plt.show()
