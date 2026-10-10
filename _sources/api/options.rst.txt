financialkit.options
====================


.. include:: /_generated/nav/options.rst

Closed-form European prices: Bachelier's normal model, Black-Scholes-Merton
with a dividend yield, Black-76 on forwards and futures, and
Garman-Kohlhagen for currencies, all broadcasting over array inputs;
put-call parity and Merton's no-arbitrage bounds; analytic Greeks checked
against finite differences, and a discretely rebalanced delta hedge on
paths from :func:`financialkit.integrators.gbm_paths`; implied volatility
(``scipy.optimize.brentq`` after a bounds check), the
Brenner-Subrahmanyam approximation, the smile and the
Breeden-Litzenberger density; Cox-Ross-Rubinstein, Rendleman-Bartter,
Leisen-Reimer and Boyle trinomial trees with American early exercise,
and the Barone-Adesi-Whaley approximation.

.. automodule:: financialkit.options
   :members:
   :undoc-members:
