"""Single authoritative specification for research outputs.

Every research table, report and the interface read units, timestamps, strategy
identifiers and metric definitions from here.  Change a definition here, not in
a script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

ANNUALIZATION_FACTOR = 252
PROTOCOL_VERSION = 3

TIMESTAMP_CONVENTION = (
    "Each row is a close-to-close return date t (Asia/Kolkata trading date, tz-naive). "
    "gross_return(t) is earned by holdings in place before close t; transaction_cost(t) is "
    "paid at close t on the trade that fills at close t; a target signalled at close t "
    "(data through t) fills at close t+1 and first earns t+1 -> t+2."
)

RETURN_UNITS = "Simple (arithmetic) daily returns as decimals; 0.01 = 1%."
COST_UNITS = "Fraction of portfolio wealth paid at the fill close; costs are one-way bps per unit of traded weight."
TURNOVER_UNITS = (
    "One-way turnover 0.5 * sum(|w_target - w_pre_trade|) per fill; the initial allocation "
    "from cash counts as sum(|w_target|) = 1.0."
)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    definition: str


METRICS: Dict[str, MetricSpec] = {m.key: m for m in [
    MetricSpec("cagr", "CAGR", "fraction/yr", "prod(1 + r_net) ** (252 / N) - 1"),
    MetricSpec("ann_vol", "Annualised volatility", "fraction/yr", "std(r_net, ddof=1) * sqrt(252)"),
    MetricSpec("sharpe", "Sharpe (rf = 0)", "ratio", "mean(r_net) / std(r_net, ddof=1) * sqrt(252); rf = 0"),
    MetricSpec("sharpe_vs_cash", "Sharpe over cash proxy", "ratio",
               "mean(r_net - r_cash) / std(r_net - r_cash, ddof=1) * sqrt(252); r_cash = LIQUIDBEES 'bond' leg"),
    MetricSpec("sortino", "Sortino (rf = 0)", "ratio",
               "mean(r_net) / sqrt(mean(min(r_net, 0)^2)) * sqrt(252); mean over all N days"),
    MetricSpec("max_drawdown", "Maximum drawdown", "fraction", "max(1 - W_t / max_{s<=t} W_s), W = cumprod(1 + r_net)"),
    MetricSpec("calmar", "Calmar", "ratio", "cagr / max_drawdown"),
    MetricSpec("ann_turnover", "Annualised turnover", "x/yr", "sum(turnover) * 252 / N"),
    MetricSpec("ann_cost", "Annualised cost", "fraction/yr", "sum(transaction_cost) * 252 / N"),
    MetricSpec("cost_drag", "Cost drag (wealth)", "fraction of initial wealth",
               "prod(1 + r_gross) - prod(1 + r_net) over the window"),
    MetricSpec("n_obs", "Observations", "days", "number of return dates in the window"),
]}

# Strategy identifiers.  IDs are stable; display names may change.
STRATEGIES: Dict[str, str] = {
    "regimeshift": "RegimeShift: 3-state HMM + regime-conditioned CVXPY optimizer",
    "no_regime_optimizer": "CVXPY optimizer with the Bull constraint set always (HMM removed)",
    "vol_rule": "63-day realised equity volatility thresholds -> fixed regime weights (no HMM)",
    "hmm_fixed": "HMM regime -> fixed regime weights (no optimizer)",
    "min_variance": "Long-only minimum variance, 252-day sample covariance, w <= 0.80",
    "static_6040": "Static 60% equity / 40% bond, rebalanced on RegimeShift fill dates",
    "equal_weight": "Static 1/3 each, rebalanced on RegimeShift fill dates",
}

EVALUATION_TYPES: Dict[str, str] = {
    "full_sample": "One continuous online backtest over the common evaluation index.",
    "chronological_attribution": (
        "A calendar slice of the SAME continuous online backtest. Holdings at the slice start "
        "were formed before it. It attributes performance to periods; it is not an independent "
        "re-fit or a walk-forward model-selection test."
    ),
    "walk_forward_selection": (
        "Each evaluation window uses a strategy chosen only from information that ended before "
        "the window started; the selection rule is fixed in advance."
    ),
}
