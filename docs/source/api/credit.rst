financialkit.credit
===================


.. include:: /_generated/nav/credit.rst

Altman's Z-score and Fisher's linear discriminant; Merton's structural
model through :func:`financialkit.options.black_scholes_price`, its
calibration to equity and the Crosbie-Bohn distance to default; Black-Cox
first passage and Leland's endogenous default; piecewise-constant hazard
curves, Jarrow-Turnbull bonds and the Duffie-Singleton, treasury and
face-value recovery conventions (with a CIR intensity through
:class:`financialkit.interest_rate_models.CIRModel`); O'Kane-Turnbull CDS
legs, par spreads, upfronts and the ISDA-style hazard bootstrap;
CreditMetrics migration and credit VaR, Jarrow-Lando-Turnbull generator
matrices (``scipy.linalg.expm`` and ``logm``) and the Israel-Rosenthal-Wei
regularization; Vasicek's large homogeneous portfolio, CreditRisk+ by
Panjer's recursion, Li's Gaussian copula (through
:func:`financialkit.risk.gaussian_copula_sample`), the Basel II IRB
formula, Hull-White tranche pricing, the Andersen-Sidenius-Basu loss
recursion and base correlation; and counterparty risk: swap exposure
profiles by Longstaff-Schwartz regression on
:class:`financialkit.interest_rate_models.HullWhiteModel` paths,
Brigo-Masetti CVA (with the swaption-strip closed form through
:func:`financialkit.interest_rate_models.jamshidian_swaption`), Gregory's
bilateral CVA and DVA, and Burgard-Kjaer funding adjustments. Every function that discounts
takes the riskless rate ``r`` as a number or as any
:class:`financialkit.fixed_income.YieldCurve`, such as a bootstrapped one.

.. automodule:: financialkit.credit
   :members:
   :undoc-members:
