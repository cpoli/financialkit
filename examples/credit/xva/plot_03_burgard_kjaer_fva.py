r"""
Burgard and Kjaer: funding valuation adjustment
===============================================

After 2008 banks could no longer borrow at the risk-free rate. An
uncollateralised trade that is in the bank's favour must be financed
until it pays, at the bank's borrowing spread, and one against it
frees cash that earns a lending spread. Burgard and Kjaer (2011)
derived the pricing PDE with both counterparty defaults and funding,
and its solution adds a funding cost and a funding benefit,

.. math:: \text{FCA} = \sum_i s_b\,EPE(t_i)\,Q_B(t_i) Q_C(t_i)\,\Delta t_i, \qquad
          \text{FBA} = \sum_i s_l\,ENE(t_i)\,Q_B(t_i) Q_C(t_i)\,\Delta t_i.

When the bank funds by issuing its own bonds, its spread is its credit
spread :math:`(1 - R_B)\lambda_B`, and the funding benefit is then the
DVA: the same cash flow counted twice if both are booked.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.credit import HazardCurve, bilateral_cva, funding_valuation_adjustment, swap_exposure
from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import HullWhiteModel, forward_swap_rate

# %%
# The funding benefit is the DVA
# ------------------------------
# A ten-year payer swap on Hull-White paths. The bank's hazard rate is 2%
# with 40% recovery, so its unsecured funding spread is 120 bp.

curve = NelsonSiegelCurve(0.04, -0.015, 0.01, 2.0)
hw = HullWhiteModel(curve, a=0.05, sigma=0.01)
K = forward_swap_rate(curve, 0.0, 10.0, 2)
prof = swap_exposure(hw, K, 10.0, notional=100.0, n_paths=4000, rng=3)

Q_bank, Q_cpty = HazardCurve.flat(0.02), HazardCurve.flat(0.03)
spread = (1 - 0.4) * 0.02
xva = bilateral_cva(prof, Q_cpty, Q_bank)
fva = funding_valuation_adjustment(prof, spread, counterparty=Q_cpty, own=Q_bank)
print(f"CVA {xva.cva:.4f}  DVA {xva.dva:.4f}")
print(f"FCA {fva.fca:.4f}  FBA {fva.fba:.4f}  FVA {fva.fva:+.4f}")
assert abs(fva.fba / xva.dva - 1) < 0.01  # equal up to the bucket rule, about lambda dt / 2

# %%
# Asymmetric funding
# ------------------
# A bank that cannot earn its borrowing spread on surplus cash has no
# funding benefit: the FVA is then the whole FCA, a cost that grows with
# the borrowing spread, and the swap is worth less to it than to a bank
# that funds at the risk-free rate.

spreads = np.linspace(0.0, 0.03, 13)
symmetric = [funding_valuation_adjustment(prof, s, counterparty=Q_cpty, own=Q_bank).fva for s in spreads]
asymmetric = [funding_valuation_adjustment(prof, s, lending_spread=0.0, counterparty=Q_cpty, own=Q_bank).fva for s in spreads]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(1e4 * spreads, symmetric, lw=2, label="borrow and lend at the same spread")
ax.plot(1e4 * spreads, asymmetric, lw=2, label="borrow at the spread, lend at the risk-free rate")
ax.axvline(1e4 * spread, color="k", ls=":", label="own credit spread")
ax.set_xlabel("borrowing spread (bp)")
ax.set_ylabel("FVA per 100 notional")
ax.legend()
assert np.all(np.diff(asymmetric) > 0) and np.all(np.array(asymmetric[1:]) > np.array(symmetric[1:]))

# %%
# The total adjustment
# --------------------
# With symmetric funding at its own credit spread, the bank's funding
# benefit and its DVA are the same cash flow, so it books one or the
# other. Either way gives the same price; booking both counts the
# bank's own default twice.

risk_free = prof.values[:, 0].mean()
with_dva = risk_free - xva.cva + xva.dva - fva.fca
with_fba = risk_free - xva.cva - fva.fva
twice = risk_free - xva.cva + xva.dva - fva.fva
print(f"risk-free value                {risk_free:+.4f}")
print(f"- CVA + DVA - FCA              {with_dva:+.4f}")
print(f"- CVA - FCA + FBA              {with_fba:+.4f}")
print(f"- CVA + DVA - FCA + FBA        {twice:+.4f}  (own default counted twice)")
assert abs(with_dva - with_fba) < 0.01 * xva.dva

plt.show()
