"""financialkit.options: vanilla option pricing, Greeks, implied volatility and lattices.

Closed forms: Bachelier's normal model, Black-Scholes-Merton with a
dividend yield, Black-76 on forwards (the one implementation the rest of
financialkit calls) and Garman-Kohlhagen for currencies, all broadcasting
over array inputs; put-call parity and Merton's no-arbitrage bounds;
analytic Greeks, checked against finite differences, and a simulated
discrete delta hedge built on :func:`financialkit.integrators.gbm_paths`.
Implied volatility by :func:`scipy.optimize.brentq` after a bounds
check, the Brenner-Subrahmanyam at-the-money shortcut, the smile, and the
Breeden-Litzenberger risk-neutral density. Lattices: Cox-Ross-Rubinstein,
Rendleman-Bartter and Leisen-Reimer binomial trees and Boyle's trinomial
tree (hand-written backward induction, with American early exercise and
the exercise boundary), plus the Barone-Adesi-Whaley approximation.
"""

from financialkit import __version__
from financialkit.options.core.base import (
    AmericanApproximationResult,
    DensityResult,
    Greeks,
    HedgeResult,
    LatticePricer,
    SmileResult,
    TreeResult,
)
from financialkit.options.systems.american import barone_adesi_whaley
from financialkit.options.systems.closed_form import bachelier_price, black76_price, black_scholes_price, garman_kohlhagen_price
from financialkit.options.systems.density import breeden_litzenberger_density
from financialkit.options.systems.greeks import black_scholes_greeks
from financialkit.options.systems.hedging import delta_hedge
from financialkit.options.systems.implied_vol import brenner_subrahmanyam_vol, implied_vol_smile, implied_volatility
from financialkit.options.systems.parity import no_arbitrage_bounds, parity_residual, put_call_parity
from financialkit.options.systems.trees import BinomialTree, CRRTree, LeisenReimerTree, RendlemanBartterTree, TrinomialTree
from financialkit.options.utils.common import black_d1_d2, finite_difference_greeks, forward_price, payoff

__all__ = [
    "__version__",
    "Greeks",
    "TreeResult",
    "AmericanApproximationResult",
    "SmileResult",
    "DensityResult",
    "HedgeResult",
    "LatticePricer",
    "bachelier_price",
    "black_scholes_price",
    "black76_price",
    "garman_kohlhagen_price",
    "put_call_parity",
    "parity_residual",
    "no_arbitrage_bounds",
    "black_scholes_greeks",
    "delta_hedge",
    "implied_volatility",
    "brenner_subrahmanyam_vol",
    "implied_vol_smile",
    "breeden_litzenberger_density",
    "BinomialTree",
    "CRRTree",
    "RendlemanBartterTree",
    "LeisenReimerTree",
    "TrinomialTree",
    "barone_adesi_whaley",
    "forward_price",
    "black_d1_d2",
    "payoff",
    "finite_difference_greeks",
]
