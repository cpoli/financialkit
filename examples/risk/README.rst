Examples
========

This gallery walks through every public feature of ``financialkit.risk``:
value at risk by historical, parametric, Cornish-Fisher, Monte Carlo,
age-weighted, filtered historical and delta-gamma methods, and adjusted
for liquidity; Roy's safety-first criterion; expected shortfall,
coherence, Euler allocation and the FRTB's liquidity horizons; extreme
value theory; VaR and ES backtests; copulas and tail dependence; and drawdowns.

See also the narrative tutorial:

- :doc:`/tutorials/from_returns_to_risk`

Each script in this gallery is self-contained and can be run directly with
``python examples/risk/<section>/<script>.py``.

Sections
--------

- **var** -- Cornish-Fisher, Roy's safety first, parametric (RiskMetrics)
  VaR, Monte Carlo VaR, hybrid VaR, filtered historical simulation,
  Rouvinez's delta-gamma VaR and liquidity-adjusted VaR.
- **expected_shortfall** -- the coherence axioms, Euler risk allocation
  the Acerbi-Tasche estimator and the FRTB liquidity-horizon cascade.
- **extreme_value** -- Fisher-Tippett block maxima and peaks over threshold.
- **backtesting** -- Kupiec's test, the Basel traffic light,
  Christoffersen's independence test and Acerbi-Szekely ES backtests.
- **copulas** -- Sklar's theorem with the Gaussian copula, and the t
  copula's tail dependence.
- **drawdowns** -- maximum drawdown and the Calmar ratio.
