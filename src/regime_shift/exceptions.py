"""
Custom exception hierarchy for the RegimeShift data pipeline.

Raised instead of generic exceptions so callers always receive:
  - which ticker or asset failed
  - the date range that was attempted
  - a human-readable reason for the failure
"""


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



