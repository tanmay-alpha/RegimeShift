# Data Directory

This directory holds multi-asset market datasets used by the RegimeShift pipeline.

## Standard Asset Universe

| Column | Asset | Default Ticker | Kind | Currency | Notes |
| :----- | :---- | :------------- | :--- | :------- | :---- |
| `equity` | NSE NIFTY 50 Index | `^NSEI` | PRICE | INR | Canonical NSE large-cap benchmark |
| `gold` | Nippon India ETF Gold Bees | `GOLDBEES.NS` | PRICE | INR | INR-denominated gold ETF (NSE); avoids USD/INR FX mismatch |
| `bond` | SBI Magnum Gilt Fund – Regular Plan | `0P0001BVE8.BO` | PRICE | INR | Indian government gilt fund NAV; genuine bond price series |
| `vix` | India VIX (preferred) | `^INDIAVIX` | INDICATOR | — | NSE implied volatility; preferred over CBOE `^VIX` |

> **Gold currency policy**: `GOLDBEES.NS` is INR-denominated and avoids USD/INR FX conversion.
> If `GC=F` (USD) is substituted, an explicit INR/USD FX adjustment is required before computing
> returns — this is NOT yet implemented in the feature-engineering phase.
>
> **VIX fallback**: `^INDIAVIX` is the preferred default. The CBOE `^VIX` (`VIX_FALLBACK` constant)
> must be set **explicitly** by overriding `TickerConfig.vix`. There is NO silent automatic fallback.

> **Yield series rejected**: `^IRX` (13-week T-Bill yield) is documented in `TickerConfig.irx_yield`
> as a reference-only `YIELD` kind spec. It cannot be used as a portfolio price; attempting to do so
> raises `TypeError` via `AssetSpec.require_price()`.

## Data Format

Files placed or cached in this directory must satisfy the contract enforced by  
[`src/regime_shift/validation.py`](../src/regime_shift/validation.py):

| Rule | Requirement |
| :--- | :---------- |
| Index type | `pd.DatetimeIndex` |
| Index order | Monotonically ascending |
| Duplicates | None |
| Timezone | Timezone-naive (UTC-normalised) |
| Required columns | `equity`, `gold`, `bond` |
| Prices | Strictly positive, finite, numeric |
| Missing values | None in required columns |
| Backward fill | **Never applied** |
| Forward fill | Configurable limit (default 3 days) |

## Workflow

### Online (live download)
```python
from regime_shift.config import RegimeShiftConfig
from regime_shift.data import download_market_data

cfg = RegimeShiftConfig()
prices = download_market_data(cfg, start="2015-01-01", include_vix=False)
```

### Offline (pre-built CSV)
```python
from regime_shift.data import load_market_data_csv

prices = load_market_data_csv("data/cache/market_data_2015-01-01_2024-12-31.csv")
```

### Saving a cache
```python
from regime_shift.data import save_market_data_csv

save_market_data_csv(prices, "data/cache/market_data_2015-01-01_2024-12-31.csv")
```

## Cache Directory

Auto-downloaded data is cached to `data/cache/` as `market_data_<start>_<end>[_vix].csv`.  
Cache files are excluded from git (see `.gitignore`).  
No database is used.

## Column Order

Output columns are **always** in the following deterministic order, regardless of
download or ingestion sequence:

```
equity  →  gold  →  bond  [→  vix]
```
