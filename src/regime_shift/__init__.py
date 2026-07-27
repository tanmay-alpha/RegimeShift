"""
RegimeShift: Market-Regime Identification & Dynamic Asset Allocation Framework.

Designed for IIT Bombay Summer Quant 2026 assignment.
"""

from regime_shift.config import (
    AssetSpec,
    DataConfig,
    FeatureConfig,
    HMMConfig,
    PortfolioConfig,
    BullConstraints,
    BearConstraints,
    CrisisConstraints,
    RegimeShiftConfig,
    SeriesKind,
    TickerConfig,
    VIX_FALLBACK,
)
from regime_shift.exceptions import (
    RegimeShiftError,
    DataDownloadError,
    DataAlignmentError,
    DataValidationError,
    FeatureEngineeringError,
    RegimeDetectionError,
    PortfolioOptimizationError,
)
from regime_shift.validation import validate_price_data, check_monotonic_index, check_no_duplicates
from regime_shift.data import download_market_data, load_market_data_csv, save_market_data_csv
from regime_shift.features import (
    compute_raw_features,
    drop_feature_warmup,
    fit_feature_scaler,
    transform_features,
    fit_transform_training_features,
)
from regime_shift.regime_model import (
    fit_hmm,
    predict_current_state,
    predict_current_probabilities,
    get_transition_matrix,
    get_state_statistics,
    get_regime,
    get_probabilities,
    RegimeSolution,
    StateStatistics,
)
from regime_shift.portfolio import (
    optimize_portfolio,
    PortfolioSolution,
)
from regime_shift.benchmarks import (
    static_60_40_weights,
    equal_weight_weights,
    validate_benchmark_weights,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "RegimeShiftConfig",
    "TickerConfig",
    "DataConfig",
    "FeatureConfig",
    "HMMConfig",
    "PortfolioConfig",
    "BullConstraints",
    "BearConstraints",
    "CrisisConstraints",
    "AssetSpec",
    "SeriesKind",
    "VIX_FALLBACK",
    # Exceptions
    "RegimeShiftError",
    "DataDownloadError",
    "DataAlignmentError",
    "DataValidationError",
    "FeatureEngineeringError",
    "RegimeDetectionError",
    "PortfolioOptimizationError",
    # Validation
    "validate_price_data",
    "check_monotonic_index",
    "check_no_duplicates",
    # Data pipeline
    "download_market_data",
    "load_market_data_csv",
    "save_market_data_csv",
    # Feature pipeline
    "compute_raw_features",
    "drop_feature_warmup",
    "fit_feature_scaler",
    "transform_features",
    "fit_transform_training_features",
    # Regime detection
    "fit_hmm",
    "predict_current_state",
    "predict_current_probabilities",
    "get_transition_matrix",
    "get_state_statistics",
    "get_regime",
    "get_probabilities",
    "RegimeSolution",
    "StateStatistics",
    # Portfolio optimization
    "optimize_portfolio",
    "PortfolioSolution",
    # Benchmarks
    "static_60_40_weights",
    "equal_weight_weights",
    "validate_benchmark_weights",
]
