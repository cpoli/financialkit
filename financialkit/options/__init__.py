"""financialkit.options: vanilla option pricing, Greeks, implied volatility and lattices.

Closed forms: Bachelier's normal model, Black-Scholes-Merton with a
dividend yield, Black-76 on forwards (the one implementation the rest of
financialkit calls) and Garman-Kohlhagen for currencies, all broadcasting
over array inputs; put-call parity and Merton's no-arbitrage bounds;
analytic Greeks, checked against finite differences, and a simulated
discrete delta hedge built on :func:`financialkit.integrators.gbm_paths`.
Haug-Haug-Lewis prices for discrete cash dividends. Implied volatility by
a vectorized safeguarded Newton solver, :func:`scipy.optimize.brentq` or
Jäckel's "Let's be rational" after a bounds check, the normal (Bachelier)
implied volatility for negative prices, the Brenner-Subrahmanyam
at-the-money shortcut, the smile, the Castagna-Mercurio vanna-volga FX
smile, and the Breeden-Litzenberger risk-neutral density. Lattices:
Cox-Ross-Rubinstein, Rendleman-Bartter and Leisen-Reimer binomial trees
and Boyle's trinomial tree (hand-written backward induction, with
American early exercise and the exercise boundary), plus the
Barone-Adesi-Whaley and Bjerksund-Stensland approximations.
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
from financialkit.options.systems.american import barone_adesi_whaley, bjerksund_stensland
from financialkit.options.systems.closed_form import bachelier_price, black76_price, black_scholes_price, garman_kohlhagen_price
from financialkit.options.systems.density import breeden_litzenberger_density
from financialkit.options.systems.dividends import discrete_dividend_price
from financialkit.options.systems.greeks import black_scholes_greeks
from financialkit.options.systems.hedging import delta_hedge
from financialkit.options.systems.implied_vol import bachelier_implied_volatility, brenner_subrahmanyam_vol, implied_vol_smile, implied_volatility
from financialkit.options.systems.lets_be_rational import normalised_black_call, normalised_implied_volatility
from financialkit.options.systems.parity import no_arbitrage_bounds, parity_residual, put_call_parity
from financialkit.options.systems.trees import BinomialTree, CRRTree, LeisenReimerTree, RendlemanBartterTree, TrinomialTree
from financialkit.options.systems.vanna_volga import fx_pillar_strikes, vanna_volga_price, vanna_volga_smile
from financialkit.options.utils.common import black_d1_d2, bump_greek, finite_difference_greeks, forward_price, payoff
from financialkit.options.visualizers.plots import (
    plot_density,
    plot_exercise_boundary,
    plot_greeks,
    plot_hedge_pnl,
    plot_option_value,
    plot_smile,
    plot_tree_convergence,
)

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
    "bachelier_implied_volatility",
    "normalised_black_call",
    "normalised_implied_volatility",
    "brenner_subrahmanyam_vol",
    "implied_vol_smile",
    "fx_pillar_strikes",
    "vanna_volga_price",
    "vanna_volga_smile",
    "breeden_litzenberger_density",
    "BinomialTree",
    "CRRTree",
    "RendlemanBartterTree",
    "LeisenReimerTree",
    "TrinomialTree",
    "barone_adesi_whaley",
    "bjerksund_stensland",
    "discrete_dividend_price",
    "forward_price",
    "black_d1_d2",
    "payoff",
    "bump_greek",
    "finite_difference_greeks",
    "plot_option_value",
    "plot_greeks",
    "plot_tree_convergence",
    "plot_exercise_boundary",
    "plot_smile",
    "plot_density",
    "plot_hedge_pnl",
]
