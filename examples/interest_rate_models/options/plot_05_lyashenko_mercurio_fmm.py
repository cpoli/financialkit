r"""
Lyashenko-Mercurio forward market model for backward-looking rates
==================================================================

LIBOR was fixed at the start of its accrual period. Its replacements are
overnight rates compounded in arrears, known only at the end of the
period. Lyashenko and Mercurio (2019) extended the forward rate
:math:`R_j(t) = [P(t, T_{j-1})/P(t, T_j) - 1]/\tau` past
:math:`T_{j-1}`: it keeps moving as the overnight fixings arrive and ends
at the compounded rate itself, a martingale under the
:math:`T_j`-forward measure throughout. Its volatility decays to zero
over the accrual period, :math:`\sigma g_j(t)`, and with a linear decay a
caplet on the compounded rate is Black's formula with total variance

.. math:: \sigma^2\left(T_{j-1} + \frac{\tau}{3}\right),

one third of the accrual period more than the LIBOR caplet.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import HullWhiteModel, fmm_caplet_price, fmm_volatility_decay, path_discount_factors, simulate_fmm_rate
from financialkit.options import implied_volatility

curve = NelsonSiegelCurve(0.04, -0.01, 0.0, 2.0)
start, end, vol = 1.0, 2.0, 0.25

# %%
# The extended forward rate freezes over the accrual period
# ---------------------------------------------------------

times, R = simulate_fmm_rate(curve, start, end, vol, 100, 20_000, rng=0)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(times, 100 * R[:12].T, lw=0.8)
axes[0].plot(times, 100 * vol * fmm_volatility_decay(times, start, end), "k--", lw=1.5, label="volatility (%)")
axes[0].axvspan(start, end, color="gray", alpha=0.15, label="accrual period")
axes[0].set_xlabel("t (years)")
axes[0].set_ylabel("R_j(t) (%)")
axes[0].set_title("Forward rate paths, lognormal FMM")
axes[0].legend(loc="upper left")

# %%
# Caplets: in arrears against forward-looking, and Monte Carlo
# ------------------------------------------------------------

K = np.linspace(0.02, 0.06, 9)
tau, P = end - start, float(curve.discount(end))
back = fmm_caplet_price(curve, start, end, K, vol)
fwd = fmm_caplet_price(curve, start, end, K, vol, backward_looking=False)
payoff = tau * P * np.maximum(R[:, -1:] - K, 0.0)
mc, se = payoff.mean(axis=0), payoff.std(axis=0, ddof=1) / np.sqrt(payoff.shape[0])
axes[1].plot(100 * K, 1e4 * back, label="compounded in arrears")
axes[1].plot(100 * K, 1e4 * fwd, label="forward-looking (LIBOR)")
axes[1].errorbar(100 * K, 1e4 * mc, yerr=1e4 * 4 * se, fmt="o", ms=3, label="Monte Carlo, 4 s.e.")
axes[1].set_xlabel("strike (%)")
axes[1].set_ylabel("caplet price (bp of notional)")
axes[1].set_title("Caplet on [1, 2]")
axes[1].legend()

# %%
# The volatility uplift, quoted LIBOR-style
# -----------------------------------------
# Quoting the in-arrears caplet as a Black volatility to the start of the
# period, as for LIBOR, gives :math:`\sigma\sqrt{1 + \tau/(3T_{j-1})}`:
# large for near periods, and finite even for the current period, which
# a LIBOR caplet would not price at all because its rate is already fixed.

starts = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
ends = starts + 0.25
caplets = fmm_caplet_price(curve, starts, ends, 0.04, vol)
F = (curve.discount(starts) / curve.discount(ends) - 1.0) / 0.25
quoted = implied_volatility(caplets / (0.25 * curve.discount(ends)), F, 0.04, starts, 0.0, 0.0)
axes[2].plot(starts, 100 * quoted, "o-", label="in-arrears caplet, quoted to T_{j-1}")
axes[2].plot(starts, 100 * vol * np.sqrt(1 + 0.25 / (3 * starts)), ":", color="k", label="sigma sqrt(1 + tau / 3T)")
axes[2].axhline(100 * vol, color="gray", ls="--", label="model sigma")
axes[2].set_xlabel("start of the 3-month period (years)")
axes[2].set_ylabel("Black volatility (%)")
axes[2].set_title("Volatility uplift")
axes[2].legend()
fig.tight_layout()

# %%
# In a short-rate model the compounded rate has the curve's forward
# -----------------------------------------------------------------
# Under Hull-White, the backward-looking rate is the compounded short
# rate. Its discounted payoff averages to
# :math:`P(0, T_{j-1}) - P(0, T_j)`, the forward the FMM starts from.

t, r = HullWhiteModel(curve, 0.1, 0.01).simulate(end, 200, 20_000, rng=0)
D = path_discount_factors(t, r)
i = np.searchsorted(t, start)
leg = D[:, -1] * (D[:, i] / D[:, -1] - 1.0)
exact = float(curve.discount(start) - curve.discount(end))
leg_se = leg.std(ddof=1) / np.sqrt(leg.size)
print(f"Hull-White compounded leg {leg.mean():.6f} +- {leg_se:.6f}, curve forward {exact:.6f}")

assert np.all(np.abs(mc - back) < 4 * se + 1e-7)  # the closed form matches simulation
assert np.all(back > fwd)
assert np.allclose(quoted, vol * np.sqrt(1 + 0.25 / (3 * starts)), atol=1e-6)
assert abs(leg.mean() - exact) < 4 * leg_se + 1e-6

plt.show()
