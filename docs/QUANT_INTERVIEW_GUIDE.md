# Quant interview guide

**Why is it causally valid?** Each close-to-close return is earned by weights installed after the prior close. The event timeline records the signal, execution and return dates separately.

**What was wrong before?** A target computed at a close was applied to the return ending at that same close, so it captured a bar before the decision could exist.

**Is it a pristine holdout?** No. Historical results were previously inspected. The repository labels the final historical segment retrospective and reserves post-dataset observations as a frozen-config lockbox.

**Why an HMM, and how is it selected?** It is a testable regime representation, not evidence of alpha. Numeric states are mapped from training features, never future returns; deterministic restarts are selected on train-window likelihood.

**What costs are included?** Per-asset one-way commission, half-spread/slippage and optional statutory assumption fields. The committed scenarios are assumptions, not measured spreads. Market impact, capacity, tracking difference and NAV-versus-traded-price effects remain unmodeled without volume and execution data.

**Is `^NSEI` tradable?** No. It is an index-level signal/benchmark series. Current equity P&L must be described as a simulation until a complete verified ETF dataset replaces it.

**What does underperformance mean?** Static benchmarks may outperform after corrected timing and costs. That negative result remains a finding, not a reason to retune the historical experiment.
