r"""
Johansen cointegration test
===========================

The Engle-Granger test regresses one series on the others and so finds at
most one cointegrating relation, and which one depends on the choice of
dependent variable. Søren Johansen (1988) estimated the whole
cointegrating space by maximum likelihood. Writing the VAR in
error-correction form, :math:`\Delta y_t = \mu + \alpha\beta' y_{t-1} +
\Gamma\Delta y_{t-1} + u_t`, the rank of :math:`\alpha\beta'` is the number
of cointegrating relations, and likelihood-ratio tests on the squared
canonical correlations between :math:`\Delta y_t` and :math:`y_{t-1}`
choose it. Here a stock, its depositary receipt and its futures contract
share one stochastic trend, so they have two cointegrating relations:
the two price gaps.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_series import engle_granger, johansen_test, simulate_arma

# %%
# Three prices driven by one trend
# --------------------------------
# The log stock price is a random walk with drift; the receipt's premium
# and the futures basis are stationary AR(1) processes. The unrestricted
# constant (``regression="c"``) allows for the drift.

rng = np.random.default_rng(1988)
n = 1000
stock = np.log(50.0) + np.cumsum(0.001 + 0.01 * rng.standard_normal(n))
premium = simulate_arma(ar=[0.9], n=n, sigma=0.003, rng=rng)
basis = simulate_arma(ar=[0.8], n=n, sigma=0.002, rng=rng)
y = np.column_stack([stock, stock + 0.01 + premium, stock + 0.005 + basis])

res = johansen_test(y, lags=1, regression="c")
print(" r   trace    5% cv   max-eig   5% cv")
for r in range(3):
    print(f" {r}  {res.trace[r]:7.2f}  {res.trace_critical_values['5%'][r]:6.2f}   {res.max_eigen[r]:7.2f}  {res.max_eigen_critical_values['5%'][r]:6.2f}")
rank = res.rank("5%")
print(f"cointegration rank: {rank}")
assert rank == 2 and res.rank("5%", test="max_eigen") == 2

# %%
# The cointegrating space contains the price gaps
# -----------------------------------------------
# Any basis of the two estimated vectors is equally valid. Projecting the
# true vectors :math:`(1, -1, 0)` and :math:`(1, 0, -1)` onto the
# estimated space shows that both lie in it.

B = res.beta[:, :2]
for v in (np.array([1.0, -1.0, 0.0]), np.array([1.0, 0.0, -1.0])):
    coef, *_ = np.linalg.lstsq(B, v, rcond=None)
    dist = np.linalg.norm(B @ coef - v) / np.linalg.norm(v)
    print(f"{v} lies in the estimated space up to a relative distance of {dist:.3f}")
    assert dist < 0.05

# Engle-Granger finds one relation per regression, and which one depends on the regressand.
eg = engle_granger(y[:, 0], y[:, 1:])
print(f"Engle-Granger of the stock on the other two: b = {eg.beta.round(2)}, reject no cointegration: {eg.rejects()}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for col, lab in zip(y.T, ("stock", "receipt", "futures"), strict=True):
    axes[0].plot(col, lw=0.8, label=lab)
axes[0].set_title("log prices: three random walks")
axes[0].set_xlabel("day")
axes[0].legend()
for i in range(2):
    axes[1].plot(y @ res.beta[:, i], lw=0.8, label=f"$\\hat\\beta_{i + 1}' y_t$")
axes[1].plot(y @ res.beta[:, 2], lw=0.8, color="0.6", label="$\\hat\\beta_3' y_t$ (rejected)")
axes[1].set_title("estimated combinations")
axes[1].set_xlabel("day")
axes[1].legend(fontsize=8)
fig.tight_layout()
plt.show()
