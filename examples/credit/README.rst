Examples
========

This gallery walks through every public feature of ``financialkit.credit``:
Altman's Z-score; Merton's structural model, Black-Cox first passage,
Leland's endogenous default and the KMV distance to default; hazard
rates and recovery conventions; CDS pricing and hazard bootstrapping;
rating migration with CreditMetrics and generator matrices; and
portfolio credit risk with Vasicek's large pool, CreditRisk+, Li's
copula, the Basel IRB formula and CDO tranches.

Each script in this gallery is self-contained and can be run directly with
``python examples/credit/<section>/<script>.py``.

Sections
--------

- **scoring** -- Altman's Z-score.
- **structural** -- Merton, Black-Cox, Leland and the distance to default.
- **reduced_form** -- Jarrow-Turnbull hazard rates and Duffie-Singleton
  recovery of market value.
- **cds** -- the CDS par spread and the ISDA hazard bootstrap.
- **migration** -- CreditMetrics, generator matrices and the
  Israel-Rosenthal-Wei regularization.
- **portfolio_loss** -- Vasicek's large pool, CreditRisk+, Li's Gaussian
  copula, the Basel IRB formula and tranche expected losses.
