Examples
========

This gallery walks through every public feature of ``financialkit.risk``:
value at risk by historical, parametric, Cornish-Fisher, Monte Carlo,
age-weighted and filtered historical methods; Roy's safety-first
criterion; expected shortfall, coherence and Euler allocation; extreme
value theory; VaR backtests; copulas and tail dependence; and drawdowns.

See also the narrative tutorial:

- :doc:`/tutorials/from_returns_to_risk`

Each script in this gallery is self-contained and can be run directly with
``python examples/risk/<section>/<script>.py``.

Sections
--------

- **var** -- Cornish-Fisher, Roy's safety first, parametric (RiskMetrics)
  VaR, Monte Carlo VaR, hybrid VaR and filtered historical simulation.
- **expected_shortfall** -- the coherence axioms, Euler risk allocation
  and the Acerbi-Tasche estimator.
- **extreme_value** -- Fisher-Tippett block maxima and peaks over threshold.
- **backtesting** -- Kupiec's test, the Basel traffic light and
  Christoffersen's independence test.
- **copulas** -- Sklar's theorem with the Gaussian copula, and the t
  copula's tail dependence.
- **drawdowns** -- maximum drawdown and the Calmar ratio.
