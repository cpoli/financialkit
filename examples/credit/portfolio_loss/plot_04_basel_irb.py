r"""
The Basel II IRB formula
========================

The Basel Committee (2004) set bank capital for credit risk with
Vasicek's model: capital per unit of exposure is the 99.9% quantile of
the large-pool loss rate, minus the expected loss already covered by
provisions, times a maturity adjustment. The supervisory asset
correlation falls from 24% to 12% as the PD rises. Gordy (2003) showed
this is the only setting where each loan's capital charge does not
depend on the rest of the portfolio.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.credit import basel_correlation, basel_irb_capital, basel_risk_weight, one_factor_loss_distribution, vasicek_quantile

pd = np.geomspace(3e-4, 0.3, 200)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for cls in ("corporate", "mortgage", "revolving", "retail"):
    ax[0].semilogx(pd, 100 * np.asarray(basel_risk_weight(pd, asset_class=cls)), label=cls)
    ax[1].semilogx(pd, 100 * np.asarray(basel_correlation(pd, cls)), label=cls)
ax[0].set_xlabel("PD")
ax[0].set_ylabel("risk weight (%), LGD 45%")
ax[0].legend()
ax[1].set_xlabel("PD")
ax[1].set_ylabel("asset correlation (%)")
print("illustrative corporate risk weights:", {f"{p:.2%}": f"{100 * basel_risk_weight(p):.2f}%" for p in (0.0003, 0.001, 0.01, 0.05, 0.2)})
for p, published in ((0.0003, 14.44), (0.001, 29.65), (0.01, 92.32), (0.2, 238.23)):
    assert abs(100 * basel_risk_weight(p) - published) < 0.005

# %%
# Capital is the gap between the 99.9% quantile and the expected loss
# -------------------------------------------------------------------

p, lgd = 0.01, 0.45
R = float(basel_correlation(p))
dist = one_factor_loss_distribution(p, R, 2000, lgd=lgd)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dist.losses, np.cumsum(dist.pmf))
ax.axvline(dist.mean, color="C0", ls=":", label=f"expected loss {dist.mean:.2%}")
ax.axvline(dist.quantile(0.999), color="C3", ls="--", label=f"99.9% quantile {dist.quantile(0.999):.2%}")
ax.set_xlim(0, 0.08)
ax.set_xlabel("portfolio loss rate")
ax.set_ylabel("cumulative probability")
ax.set_title(f"K at M = 1: {basel_irb_capital(p, lgd, maturity=1.0):.2%} of exposure")
ax.legend()
K = basel_irb_capital(p, lgd, maturity=1.0)
assert np.isclose(K, lgd * (vasicek_quantile(0.999, p, R) - p))
assert abs((dist.quantile(0.999) - dist.mean) / K - 1) < 0.05

plt.show()
