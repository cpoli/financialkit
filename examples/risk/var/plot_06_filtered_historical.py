r"""
Barone-Adesi: filtered historical simulation
============================================

Historical simulation keeps the empirical shape of the tail but not its
current scale; a GARCH model tracks the scale but imposes a shape.
Barone-Adesi, Giannopoulos and Vosper (1999) combined them: divide each
past return by that day's volatility to get standardized residuals
:math:`z_t = r_t / \sigma_t`, then rescale them by tomorrow's forecast,

.. math:: \text{scenarios} = \hat\sigma_{n+1}\, z_t .

The fat tail of the residuals is kept while the VaR rises and falls
with volatility.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.risk import christoffersen_test, filtered_historical_var, historical_var, kupiec_test, var_exceptions
from financialkit.risk.visualizers import plot_var
from financialkit.time_series import GARCH

# %%
# A GARCH market with Student-t shocks
# ------------------------------------
#
# The data follow a GARCH(1,1) recursion driven by fat-tailed Student-t
# shocks, the case filtered historical simulation was designed for.

rng = np.random.default_rng(5)
n, omega, a, b, nu = 2500, 1e-6, 0.08, 0.9, 5
z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
sig2 = np.empty(n)
r = np.empty(n)
sig2[0] = omega / (1 - a - b)
for t in range(n):
    if t:
        sig2[t] = omega + a * r[t - 1] ** 2 + b * sig2[t - 1]
    r[t] = np.sqrt(sig2[t]) * z[t]

# %%
# Estimating the filter
# ---------------------
#
# As in practice, the filter is estimated: a Gaussian quasi-likelihood
# GARCH(1,1) fit from :mod:`financialkit.time_series` on the first 1,000
# days. Its conditional variances use only past returns, so the
# forecasts below are out of sample from day 1,000 on.

est = 1000
model = GARCH.fit(r[:est]).model
print(f"fitted {model} (true omega {omega:.1e}, alpha {a}, beta {b})")
sigma = np.sqrt(np.r_[model.variance(r), model.forecast(r, 1)])

# %%
# Rolling one-day forecasts
# -------------------------
#
# Both methods are exceeded about as often as they should be, but
# historical simulation's exceptions arrive in clusters after volatility
# rises, which Christoffersen's conditional-coverage test detects.

window = 500
days = np.arange(est, n)
p_uc, p_cc = {}, {}
fhs = np.array([filtered_historical_var(r[t - window : t], sigma[t - window : t], sigma[t], 0.99).var for t in days])
hs = np.array([historical_var(r[t - window : t], 0.99).var for t in days])
for name, v in [("historical", hs), ("filtered historical", fhs)]:
    hits = var_exceptions(r[est:], v)
    uc, cc = kupiec_test(hits), christoffersen_test(hits)
    print(f"{name:20s}: {uc.n_exceptions} exceptions (expected {uc.expected:.0f}), Kupiec p = {uc.p_value:.3f}, conditional coverage p = {cc.p_value:.3f}")
    p_uc[name], p_cc[name] = uc.p_value, cc.p_value
assert min(p_uc.values()) > 0.05 and p_cc["historical"] < 0.05 < p_cc["filtered historical"]

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(days, r[est:], color="0.75", lw=0.6)
ax.plot(days, -hs, label="historical")
ax.plot(days, -fhs, label="filtered historical")
ax.set_xlabel("day")
ax.legend(loc="lower left")
fig.tight_layout()

# %%
# Tomorrow's filtered scenario distribution
# -----------------------------------------

plot_var(filtered_historical_var(r[-window:], sigma[n - window : n], sigma[n], 0.99))

plt.show()
