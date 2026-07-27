"""
Test configuration parameters and defaults.

Covers both the pipeline configuration (RegimeShiftConfig) and the new
asset-spec/ticker configuration (TickerConfig with AssetSpec).
"""

from regime_shift.config import (
    AssetSpec,
    RegimeShiftConfig,
    SeriesKind,
    TickerConfig,
    VIX_FALLBACK,
)


def test_config_defaults():
    """Verify pipeline parameters meet IITB assignment requirements."""
    config = RegimeShiftConfig()

    assert config.annualization_factor == 252, (
        f"Annualization factor must be 252 for Indian equity/bond markets, "
        f"got {config.annualization_factor}"
    )
    assert 5.0 <= config.transaction_cost_bps <= 10.0, (
        f"Transaction costs must be between 5 and 10 bps, "
        f"got {config.transaction_cost_bps}"
    )
    assert config.number_of_regimes == 3, (
        f"Number of regimes must be 3 (Bull, Bear, Crisis), "
        f"got {config.number_of_regimes}"
    )
    assert config.equity_col == "equity"
    assert config.gold_col == "gold"
    assert config.bond_col == "bond"


def test_default_ticker_config_all_price_assets():
    """All required asset specs in the default TickerConfig must be PRICE kind."""
    cfg = TickerConfig()
    cfg.validate_price_assets()  # must not raise


def test_default_bond_is_price_not_yield():
    """Default bond ticker must be a PRICE series (NAV), not a YIELD/rate series."""
    cfg = TickerConfig()
    assert cfg.bond.kind is SeriesKind.PRICE, (
        f"Default bond spec must be PRICE, got {cfg.bond.kind}. "
        "^IRX (yield) must NOT be the default."
    )
    assert cfg.bond.ticker != "^IRX", (
        "Default bond ticker must not be ^IRX (T-Bill yield — a rate, not a NAV price)."
    )


def test_default_gold_is_inr_price():
    """Default gold spec must be an INR-denominated PRICE series."""
    cfg = TickerConfig()
    assert cfg.gold.kind is SeriesKind.PRICE
    assert cfg.gold.currency == "INR", (
        f"Default gold spec must be INR-denominated; got currency='{cfg.gold.currency}'. "
        "GC=F (USD) requires FX conversion and must not be the default."
    )


def test_default_vix_is_india_vix():
    """Default VIX spec must be ^INDIAVIX, not CBOE VIX (^VIX)."""
    cfg = TickerConfig()
    assert cfg.vix.ticker == "^INDIAVIX", (
        f"Preferred VIX is ^INDIAVIX; got '{cfg.vix.ticker}'. "
        "Use VIX_FALLBACK for explicit CBOE fallback."
    )


def test_vix_fallback_constant_defined():
    """VIX_FALLBACK must be the CBOE ^VIX ticker string."""
    assert VIX_FALLBACK == "^VIX"


def test_irx_yield_reference_retained():
    """^IRX must remain accessible as a YIELD reference spec (not deleted)."""
    cfg = TickerConfig()
    assert cfg.irx_yield.ticker == "^IRX"
    assert cfg.irx_yield.kind is SeriesKind.YIELD


def test_asset_spec_require_price_on_yield_raises():
    """AssetSpec.require_price() must raise TypeError for YIELD kind."""
    spec = AssetSpec(ticker="^IRX", kind=SeriesKind.YIELD, description="yield")
    try:
        spec.require_price()
        assert False, "Expected TypeError"
    except TypeError as exc:
        assert "yield" in str(exc).lower()


def test_ticker_to_col_uses_configured_tickers():
    """ticker_to_col must map configured ticker symbols to output column names."""
    config = RegimeShiftConfig()
    mapping = config.ticker_to_col
    assert config.tickers.equity_ticker in mapping
    assert mapping[config.tickers.equity_ticker] == config.equity_col


def test_col_to_ticker_is_inverse_of_ticker_to_col():
    """col_to_ticker must be the exact inverse of ticker_to_col."""
    config = RegimeShiftConfig()
    for ticker, col in config.ticker_to_col.items():
        assert config.col_to_ticker[col] == ticker


# ---------------------------------------------------------------------------
# HMM config propagation through the backtest
# ---------------------------------------------------------------------------

def test_hmm_config_propagates_to_fit_hmm(monkeypatch):
    """A custom HMMConfig on RegimeShiftConfig must reach fit_hmm."""
    from regime_shift import backtest as backtest_module
    from regime_shift.config import HMMConfig
    from regime_shift.regime_model import RegimeSolution
    import pandas as pd
    import numpy as np

    config = RegimeShiftConfig()
    custom_hmm = HMMConfig(n_iter=17, tolerance=1e-5, random_state=123)
    config.hmm_config = custom_hmm

    seen_configs = []

    def fake_fit_hmm(scaled, raw, config):
        seen_configs.append(config)
        from hmmlearn.hmm import GaussianHMM
        hmm = GaussianHMM(n_components=3, covariance_type="diag",
                          n_iter=17, tol=1e-5, random_state=123)
        hmm.fit(scaled.values)
        hmm._state_map = {0: "Bull", 1: "Bear", 2: "Crisis"}
        hmm._state_statistics = []
        return hmm, np.zeros(len(scaled), dtype=int), pd.DataFrame(
            np.eye(3),
            index=["Bull", "Bear", "Crisis"],
            columns=["Bull", "Bear", "Crisis"],
        )

    def fake_predict_current_state(hmm, scaler, raw, raw_train, config):
        seen_configs.append(config)
        probs = pd.Series([1.0, 0.0, 0.0], index=["Bull", "Bear", "Crisis"])
        return RegimeSolution(
            regime="Bull",
            probabilities=probs,
            transition_matrix=pd.DataFrame(
                np.eye(3),
                index=["Bull", "Bear", "Crisis"],
                columns=["Bull", "Bear", "Crisis"],
            ),
            state_statistics=[],
            convergence=True,
            n_iter=17,
            log_likelihood=0.0,
        )

    monkeypatch.setattr(backtest_module, "fit_hmm", fake_fit_hmm)
    monkeypatch.setattr(backtest_module, "predict_current_state", fake_predict_current_state)

    # Build deterministic synthetic prices
    idx = pd.bdate_range("2015-01-01", periods=600)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({
        "equity": 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, 600))),
        "gold": 1800.0 + np.cumsum(rng.normal(0.0001, 0.006, 600)),
        "bond": 97.0 + np.cumsum(rng.normal(0.00005, 0.002, 600)),
    }, index=idx)
    config.minimum_training_observations = 60
    backtest_module.run_walk_forward_backtest(prices, config)

    assert seen_configs, "fit_hmm / predict_current_state were not called"
    for seen in seen_configs:
        assert seen.n_iter == 17, (
            f"Custom HMMConfig.n_iter did not propagate: got {seen.n_iter}"
        )
        assert seen.tolerance == 1e-5
        assert seen.random_state == 123
