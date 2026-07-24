# Data Directory

This directory is designated for multi-asset market datasets used by the RegimeShift asset allocation system.

## Standard Asset Universe

1. **NSE Equity Index** (e.g., NIFTY 50 / `^NSEI`)
2. **Gold** (e.g., MCX Gold / Gold ETF / `GC=F`)
3. **Indian Sovereign Bonds** (e.g., 10-Year G-Sec / `TLT` proxy)
4. **Volatility Index (Optional)** (e.g., India VIX / `^INDIAVIX`)

## Data Format Requirements

Files placed in this directory must meet the data contract enforced by `regime_shift.validation.validate_price_data`:
- CSV format with `Date` column as index (`DatetimeIndex`)
- Monotonic, strictly ascending dates
- No duplicate timestamps
- Columns matching asset names (`equity`, `gold`, `bond`, optional `vix`)
- Strictly positive price values
- Zero missing/infinite values
