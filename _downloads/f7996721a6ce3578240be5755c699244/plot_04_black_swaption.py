r"""
Jamshidian's swap market model: Black swaptions
===============================================

Jamshidian (1997) showed that the market's way of quoting swaptions is
also exact in a model of its own. Under the measure that uses the
swap's fixed-leg annuity :math:`A = \sum_i \tau_i P(0, t_i)` as
numeraire, the forward swap rate
:math:`S = [P(0, T_0) - P(0, T_n)]/A` is a martingale, and if it is
lognormal a payer swaption is

.. math:: V = A\,[S N(d_1) - K N(d_2)],

Black's formula on the swap rate, times the annuity. The swap market
model and the LIBOR market model cannot both be lognormal, but each
prices its own instruments exactly.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import HullWhiteModel, black_swaption_price, forward_swap_annuity, forward_swap_rate, jamshidian_swaption
from financialkit.options import implied_volatility

# %%
# Hull's example
# --------------
#
# Hull (Example 29.5): a 5-year option to pay 6.2% on a 3-year
# semiannual swap, curve flat at 6% continuously compounded, 20%
# volatility, principal 100.

print(f"payer swaption: {black_swaption_price(0.06, 5.0, 3.0, 0.062, 0.2, 2, notional=100.0):.2f} (Hull: 2.07)")

# %%
# Payer, receiver and parity
# --------------------------

curve = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)
expiry, tenor = 3.0, 5.0
S, A = forward_swap_rate(curve, expiry, tenor), forward_swap_annuity(curve, expiry, tenor)
K = np.linspace(0.02, 0.09, 50)
payer = black_swaption_price(curve, expiry, tenor, K, 0.2)
receiver = black_swaption_price(curve, expiry, tenor, K, 0.2, payer=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(100 * K, payer, label="payer")
axes[0].plot(100 * K, receiver, label="receiver")
axes[0].plot(100 * K, payer - receiver, "k:", label="payer - receiver = A (S - K)")
axes[0].axvline(100 * S, color="0.6", lw=0.8)
axes[0].set_xlabel("fixed rate (%)")
axes[0].set_ylabel("price per unit notional")
axes[0].legend()
print(f"max parity error: {np.max(np.abs(payer - receiver - A * (S - K))):.1e}")

# %%
# The Black volatilities a Hull-White model quotes
# ------------------------------------------------
#
# Price swaptions exactly in Hull-White by Jamshidian's decomposition,
# then invert Black's formula: the model's volatility smile in swap-rate
# terms is downward sloping, the signature of a normal (Gaussian) model.

hw = HullWhiteModel(curve, 0.1, 0.01)
for tenor_ in (2.0, 5.0, 10.0):
    S_, A_ = forward_swap_rate(curve, expiry, tenor_), forward_swap_annuity(curve, expiry, tenor_)
    Ks = S_ + np.linspace(-0.02, 0.02, 9)
    prices = np.array([jamshidian_swaption(hw, expiry, tenor_, k).price for k in Ks])
    vols = implied_volatility(prices / A_, S_, Ks, expiry, 0.0, 0.0)
    axes[1].plot(100 * (Ks - S_), 100 * vols, "o-", ms=3, label=f"3y x {tenor_:g}y")
axes[1].set_xlabel("strike - forward swap rate (%)")
axes[1].set_ylabel("Black swaption volatility (%)")
axes[1].legend(title="Hull-White a = 0.1, sigma = 1%")
fig.tight_layout()

plt.show()
