r"""
G2++ calibrated to a swaption grid
==================================

A one-factor model moves the whole curve with one shock, so a swaption
on a 1-year swap and one on a 10-year swap see the same source of
risk; the model has nothing to trade off between short and long tenors.
G2++'s two correlated factors do. Swaptions have no Jamshidian
decomposition in two factors, but Brigo and Mercurio (2001) reduce the
price to a one-dimensional integral: given the first factor at expiry,
the swap is in the money on one side of a single level of the second,
so the conditional price is a closed form in normal CDFs.

Here a grid of ATM swaption volatilities is quoted from a known G2++
model, and both Hull-White and G2++ are fitted to it by vega-weighted
least squares.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import G2PlusPlusModel, g2pp_calibrate, hull_white_calibrate, model_implied_vols

curve = NelsonSiegelCurve(0.04, -0.015, 0.01, 2.0)
truth = G2PlusPlusModel(curve, a=0.6, sigma=0.012, b=0.06, eta=0.007, rho=-0.6)

expiry_axis, tenor_axis = np.array([1.0, 2.0, 5.0, 10.0]), np.array([1.0, 5.0, 10.0])
expiries, tenors = (g.ravel() for g in np.meshgrid(expiry_axis, tenor_axis))

# %%
# Quotes from the known model
# ---------------------------
# :func:`~financialkit.interest_rate_models.model_implied_vols` prices each
# swaption with :func:`~financialkit.interest_rate_models.g2pp_swaption`
# and inverts Black's formula.

vols = model_implied_vols(truth, expiries, "swaption", tenors)

# %%
# Two fits
# --------
# G2++ recovers the five parameters; Hull-White's two cannot match the
# short and long tenors at once.

g2 = g2pp_calibrate(curve, vols, expiries, kind="swaption", tenors=tenors)
hw = hull_white_calibrate(curve, vols, expiries, "swaption", tenors)
print("G2++ fit:  " + ", ".join(f"{k} = {v:.4f}" for k, v in g2.params.items()) + f"; RMSE = {100 * g2.rmse:.1e} vol points")
print(f"Hull-White: a = {hw.params['a']:.4f}, sigma = {hw.params['sigma']:.5f}; RMSE = {100 * hw.rmse:.2f} vol points")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, fit, name in zip(axes, (g2, hw), ("G2++", "Hull-White"), strict=True):
    for i, (tenor, color) in enumerate(zip(tenor_axis, ("C0", "C1", "C2"), strict=True)):
        rows = slice(i * expiry_axis.size, (i + 1) * expiry_axis.size)
        ax.plot(expiry_axis, 100 * vols[rows], "o", color=color, label=f"market, {tenor:g}y swap")
        ax.plot(expiry_axis, 100 * fit.model_vols[rows], "-", color=color)
    ax.set_xlabel("swaption expiry (years)")
    ax.set_title(f"{name} fit (lines), RMSE {100 * fit.rmse:.2g} vol points")
axes[0].set_ylabel("ATM Black volatility (%)")
axes[0].legend()
fig.tight_layout()

assert g2.rmse < 1e-8  # exact quotes are fitted exactly
true_params = {"a": 0.6, "sigma": 0.012, "b": 0.06, "eta": 0.007, "rho": -0.6}
assert all(abs(g2.params[k] / v - 1) < 1e-4 for k, v in true_params.items())  # and identify all five parameters
assert hw.rmse > 1000 * g2.rmse and hw.rmse > 1e-3  # one factor misses by more than a tenth of a vol point

plt.show()
