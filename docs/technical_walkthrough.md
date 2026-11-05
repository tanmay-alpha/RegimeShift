# RegimeShift technical walkthrough

## 1. Start with the hypothesis, not the algorithm

Question: does conditioning allocation on estimated market regimes improve risk-adjusted outcomes after costs, compared with fixed allocation and simpler rules?

A useful answer can be no. Equal weight beating a complicated model is evidence about estimation error, turnover and whether the regime labels carry decision-relevant information. Positive CAGR alone is not alpha: the underlying equity and gold markets can rise without any model skill.

The project is a three-asset historical index-level study. It does not establish a tradeable index fill, complete ETF total returns or live profitability.

## 2. Features and information boundaries

Momentum describes how far the price moved over a trailing window. Realized volatility measures variation in trailing returns. Correlation describes co-movement, not causation. The feature scaler learns its means and variances from the current training window only: fitting it on the complete historical sample would leak later distribution information.

Every decision at close t uses only rows through t. However, the historical source repair looks ahead to identify a spike that reverses. That is a separate, ex-post data assumption. Distinguish **trailing-only model logic** from **point-in-time source data**; only the former is established here.

## 3. HMM in plain language

An HMM assumes an unobserved state z_t produces the observed feature vector x_t. State transitions follow a matrix A, with A_ij = P(z_t=j | z_(t-1)=i). Conditional on a state, this model uses a Gaussian feature distribution with diagonal covariance.

Filtering combines the previous state belief and transition matrix with the likelihood of today's observed features:

`p(z_t | x_1:t) ∝ p(x_t | z_t) × sum_i A_i,z_t p(z_(t-1)=i | x_1:t-1)`.

The engine reads only the final posterior of a sequence ending at t. Earlier posteriors smoothed using later rows would not have been available on their historical dates. Training likelihood chooses among deterministic EM restarts; later portfolio performance must not choose the restart.

Numeric states 0, 1 and 2 have no economic identity. Training-period volatility and momentum map them to Bull/Bear/Crisis interpretations. Relabelled clusters can still change character between fits. Overlap-adjusted partition stability is more informative than comparing raw state IDs.

A high Crisis posterior says the current features resemble a fitted latent state. It is not the probability of a future crash. To claim crash forecasting, define a future-return target and horizon, purge overlapping labels, fit only past examples, then score probabilities out of sample with Brier/log loss and calibration diagnostics against a base-rate forecast. That task has not been established here.

## 4. Uncertainty, duration and diagnostics

Entropy `-sum(p log p)` rises when the state belief is spread out; dividing by log(3) gives a 0–1 scale. It is model-conditional uncertainty, not uncertainty about whether the whole HMM is misspecified. An exploratory entropy threshold is an alert rule, not a validated risk predictor.

Under a fixed transition matrix, expected persistence in state i is `1 / (1 - A_ii)` observations. An absorbing state gives infinite expected duration. Actual observed regime spells differ because parameters are refitted and signal labels are sampled periodically. The final spell is right-censored: its end was not observed, so do not treat it as a completed duration.

State feature means describe clusters. They are not causal feature contributions or proof that changing a feature would change a portfolio outcome.

## 5. Portfolio mathematics

A long-only portfolio has weights w_i >= 0 with sum(w)=1. Estimated variance is `w' Sigma w`. Minimum variance minimizes this expression under concentration caps. Expected returns are noisy; adding a return-reward term can make allocations unstable even when covariance estimates appear precise.

The original optimizer changes feasible sets by regime: Bull permits more equity, Crisis demands defensive weights. This embeds economic assumptions. Removing the HMM while keeping one constraint set tests a different assumption from removing the optimizer while retaining fixed regime weights; neither ablation alone proves causal attribution.

Ledoit–Wolf covariance shrinks noisy sample covariance toward a structured target. Risk budgeting solves a positive convex program with log-weight penalties, normalizes the solution and verifies Euler risk contributions. Exact equal-risk contribution is not the same as inverse volatility when assets are correlated.

Variance contribution of asset i is `w_i (Sigma w)_i`; dividing by total variance gives its signed share. Negative contributions can be legitimate hedging effects. Clipping a risk-parity solution to a weight cap generally destroys equal contributions; the lab documents this constraint difference instead of hiding it.

Volatility targeting blends an unlevered reference portfolio with the defensive leg. The defensive leg is not mathematically zero-risk; use the full covariance quadratic when solving for the blend. Turnover-aware allocation trades a measured reduction in changing weights against estimated variance. Neither a risk target nor an optimization solution guarantees realized risk.

## 6. Trace a trade through time

At close t, the old holdings first earn the return ending t. A pending target formed at t-1 trades at this close and pays cost. Data through t then produce a new target, which cannot fill until t+1. That target first earns the return from t+1 to t+2.

Gross return is the old weight vector dotted with asset returns. Weights drift with relative asset performance. Cost is computed from actual differences between target and drifted weights, not the previous target. Net return is `(1 + gross) * (1 - cost) - 1`.

Initial allocation from cash is a real trade: gross is zero on its fill date and entry cost is charged. A standalone walk-forward selector must not inherit the chosen candidate's already-funded book for free. Calendar attribution is different: it intentionally inherits the continuous portfolio and does not pretend to restart it.

## 7. Metrics and fair comparison

CAGR compounds simple daily returns and annualizes using 252/N. Volatility uses sample standard deviation times sqrt(252). Sharpe is mean excess return divided by its standard deviation times sqrt(252). rf=0 rewards cash carry; the lab also subtracts daily returns of the defensive proxy. An incomplete cash proxy limits that interpretation.

Sortino uses the square root of the mean of `min(excess,0)^2` **over every day**, including zero contributions on non-downside days. Conditioning its denominator only on negative days changes the definition.

Drawdown compares wealth with its running peak including initial capital 1.0. Starting the peak at the first post-return wealth hides a first-day loss. Maximum drawdown is reported as a positive loss magnitude; plotted drawdown is usually negative.

Cost drag is the difference between compounded gross and net final wealth, in units of initial wealth. It is not the sum of cost rates, an annual fee, or percentage points of CAGR. Turnover convention counts the initial buy as 1 and subsequent one-way turnover as half the L1 asset-weight change; asset trading costs use the full absolute change across both buys and sells.

## 8. Chronological validation and uncertainty

A calendar slice of one online backtest attributes performance to that period. It is not a new independent experiment. The selection experiment ranks candidates on a trailing validation window ending before an evaluation year, then executes the selection on the next close. All candidates must face the same evaluation dates and funding assumptions.

Paired bootstrap resamples the same contiguous day blocks for every strategy. This preserves cross-strategy pairing and some short-range dependence. Different block lengths test sensitivity; none guarantees stationarity over a changing market. The family of hypotheses must be declared, and Holm adjustment controls family-wise error among the reported tests. A marginal 95% interval excluding zero is not enough after examining many comparisons.

DSR penalizes a Sharpe estimate for non-normal moments and selection among trials, under approximate IID and independent-trial assumptions. Correlated strategies, serial dependence and missing historical trial counts prevent a definitive probability-of-skill interpretation here.

CSCV asks whether the best trial on half the contiguous blocks ranks poorly on the complement. It deliberately mixes chronological blocks and therefore cannot replace chronological holdout evaluation. Undefined candidate scores must fail, not silently become reassuring PBO=0.

## 9. Stress, prospective evidence and interview answers

Historical replay asks what the portfolio actually did during a past episode with inherited holdings. Hypothetical shocks ask how a specified book responds under stated price/correlation assumptions. They are not forecasts. Close-only data do not support invented overnight gaps, spreads or capacity.

A useful freeze binds code, configuration, training policy, data prefix and environment before accepting later observations. A local file hash detects accidental mutation but does not independently prove a historical timestamp. Existing results remain retrospective. A true prospective claim needs independently verifiable pre-observation registration, immutable data vintages and a record of every attempted evaluation.

**Why not deep learning?** The scientific question is whether regime information adds value. Simpler controls already challenge the HMM; more flexible models would increase selection burden without fixing source quality.

**What is the strongest engineering decision?** One event engine and one accounting contract serve all strategies, CLI and UI. A comparison fails on missing dates rather than manufacturing zero returns.

**What would improve the study most?** Verified corporate actions and distribution-complete instrument data; independently timestamped prospective protocols; matched-risk comparisons; uncertainty-aware budgets tested against fixed weights, not merely more model complexity.
