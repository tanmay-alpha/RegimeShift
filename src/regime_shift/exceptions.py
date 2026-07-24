from typing import Optional


class RegimeShiftError(Exception):
    """Base exception for all RegimeShift errors."""


class DataValidationError(RegimeShiftError, ValueError):
    """
    Raised when a loaded DataFrame violates the data contract.

    Inherits from ValueError so that callers using ``pytest.raises(ValueError)``
    continue to work without changes.

    Attributes:
        reason: Human-readable description of the violated rule.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Data validation failed: {reason}")


class DataDownloadError(RegimeShiftError):
    """
    Raised when a market data download fails or returns an empty result.

    Attributes:
        ticker: The yfinance ticker symbol that failed.
        start: Requested start date string.
        end: Requested end date string.
        reason: Human-readable failure explanation.
    """

    def __init__(self, ticker: str, start: str, end: str, reason: str) -> None:
        self.ticker = ticker
        self.start = start
        self.end = end
        self.reason = reason
        super().__init__(
            f"Failed to download '{ticker}' ({start} → {end}): {reason}"
        )


class DataAlignmentError(RegimeShiftError):
    """
    Raised when assets cannot be aligned to a common set of trading dates.

    Attributes:
        reason: Human-readable explanation of the alignment failure.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Asset alignment failed: {reason}")


class FeatureEngineeringError(RegimeShiftError, ValueError):
    """
    Raised when feature computation encounters an error or produces invalid output.

    Inherits from ValueError for compatibility with pytest.raises(ValueError).

    Attributes:
        reason: Human-readable description of the failure.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Feature engineering failed: {reason}")


class RegimeDetectionError(RegimeShiftError, ValueError):
    """
    Raised when HMM regime detection encounters an error.

    Inherits from ValueError for compatibility with pytest.raises(ValueError).

    Attributes:
        reason: Human-readable description of the failure.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Regime detection failed: {reason}")


class PortfolioOptimizationError(RegimeShiftError, ValueError):
    """
    Raised when portfolio optimization fails or produces invalid weights.

    Inherits from ValueError for compatibility with pytest.raises(ValueError).

    Attributes:
        reason: Human-readable description of the failure.
        regime: The regime label that triggered the failure.
        solver: The solver that was attempted.
        solver_status: The status returned by the solver.
    """

    def __init__(
        self,
        reason: str,
        regime: Optional[str] = None,
        solver: Optional[str] = None,
        solver_status: Optional[str] = None,
    ) -> None:
        self.reason = reason
        self.regime = regime
        self.solver = solver
        self.solver_status = solver_status
        parts = [f"Portfolio optimization failed: {reason}"]
        if regime:
            parts.append(f"Regime: {regime}")
        if solver:
            parts.append(f"Solver: {solver}")
        if solver_status:
            parts.append(f"Solver status: {solver_status}")
        super().__init__(" | ".join(parts))



