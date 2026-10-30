"""financialkit.market_microstructure: spreads, dealers, information, trade impact and execution.

Demsetz's quoted, effective and realized spreads, and the Roll and
Corwin-Schultz estimators that need no quotes; Lee-Ready trade
classification; Garman's Poisson dealer, Ho-Stoll inventory quotes and
Avellaneda-Stoikov market making (simulated around a Brownian mid from
:mod:`financialkit.integrators`); Kyle's insider and Glosten-Milgrom's
dealer; Hasbrouck's trade-impact VAR; PIN by maximum likelihood and
VPIN by volume buckets; the square-root impact law, Bertsimas-Lo and
Almgren-Chriss optimal execution and Obizhaeva-Wang transient impact;
Amihud illiquidity and Kyle's lambda by regression; Hawkes self-exciting
order flow, the Cont-Stoikov-Talreja order book and Cont-Kukanov-Stoikov
order-flow imbalance. Trade signs are +1 for buys and -1 for sells.
"""

from financialkit import __version__
from financialkit.market_microstructure.core.base import (
    BertsimasLoResult,
    ExecutionResult,
    GarmanResult,
    GlostenMilgromResult,
    HawkesResult,
    KyleResult,
    MarketMakingResult,
    OrderBookResult,
    PINResult,
    QuotingStrategy,
    TradeImpactResult,
    TransientImpactResult,
    VPINResult,
)
from financialkit.market_microstructure.systems.classification import lee_ready, quote_rule, tick_test
from financialkit.market_microstructure.systems.dealers import (
    AvellanedaStoikov,
    SymmetricQuotes,
    avellaneda_stoikov_spread,
    garman_ruin_probability,
    ho_stoll_quotes,
    reservation_price,
    simulate_garman,
    simulate_market_making,
)
from financialkit.market_microstructure.systems.execution import (
    almgren_chriss,
    bertsimas_lo,
    execution_cost,
    fit_impact_law,
    obizhaeva_wang,
    square_root_impact,
    transient_impact_cost,
)
from financialkit.market_microstructure.systems.information import glosten_milgrom_quotes, kyle_equilibrium, simulate_glosten_milgrom, simulate_kyle
from financialkit.market_microstructure.systems.informed_trading import bulk_volume_classification, fit_pin, pin_loglik, simulate_pin, vpin
from financialkit.market_microstructure.systems.liquidity import amihud_illiquidity, kyle_lambda
from financialkit.market_microstructure.systems.order_flow import (
    cst_mid_up_probability,
    fit_hawkes,
    fit_ofi_impact,
    hawkes_compensator,
    hawkes_intensity,
    hawkes_loglik,
    order_flow_imbalance,
    simulate_cst,
    simulate_hawkes,
)
from financialkit.market_microstructure.systems.spreads import (
    corwin_schultz_spread,
    effective_spread,
    price_impact,
    quoted_spread,
    realized_spread,
    roll_spread,
)
from financialkit.market_microstructure.systems.trade_impact import hasbrouck_var
from financialkit.market_microstructure.utils.quotes import align_quotes, midquote, trade_signs
from financialkit.market_microstructure.visualizers.plots import (
    plot_execution,
    plot_garman,
    plot_glosten_milgrom,
    plot_impulse_response,
    plot_market_making,
    plot_quotes,
    plot_vpin,
)

__all__ = [
    "__version__",
    "KyleResult",
    "GlostenMilgromResult",
    "GarmanResult",
    "MarketMakingResult",
    "ExecutionResult",
    "BertsimasLoResult",
    "TradeImpactResult",
    "PINResult",
    "VPINResult",
    "HawkesResult",
    "OrderBookResult",
    "TransientImpactResult",
    "QuotingStrategy",
    "quoted_spread",
    "effective_spread",
    "realized_spread",
    "price_impact",
    "roll_spread",
    "corwin_schultz_spread",
    "tick_test",
    "quote_rule",
    "lee_ready",
    "garman_ruin_probability",
    "simulate_garman",
    "ho_stoll_quotes",
    "reservation_price",
    "avellaneda_stoikov_spread",
    "AvellanedaStoikov",
    "SymmetricQuotes",
    "simulate_market_making",
    "kyle_equilibrium",
    "simulate_kyle",
    "glosten_milgrom_quotes",
    "simulate_glosten_milgrom",
    "hasbrouck_var",
    "pin_loglik",
    "fit_pin",
    "simulate_pin",
    "bulk_volume_classification",
    "vpin",
    "square_root_impact",
    "fit_impact_law",
    "bertsimas_lo",
    "almgren_chriss",
    "execution_cost",
    "obizhaeva_wang",
    "transient_impact_cost",
    "amihud_illiquidity",
    "kyle_lambda",
    "simulate_hawkes",
    "hawkes_intensity",
    "hawkes_loglik",
    "hawkes_compensator",
    "fit_hawkes",
    "simulate_cst",
    "cst_mid_up_probability",
    "order_flow_imbalance",
    "fit_ofi_impact",
    "midquote",
    "align_quotes",
    "trade_signs",
    "plot_quotes",
    "plot_glosten_milgrom",
    "plot_garman",
    "plot_market_making",
    "plot_execution",
    "plot_impulse_response",
    "plot_vpin",
]
