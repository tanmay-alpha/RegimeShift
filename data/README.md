# Data Directory

This directory holds multi-asset market datasets used by the RegimeShift pipeline.

## Standard Asset Universe

| Column | Asset | Default Ticker | Source | Rationale |
| :----- | :---- | :------------- | :----- | :-------- |
| `equity` | NSE NIFTY 50 Index | `^NSEI` | Yahoo Finance | Canonical NSE large-cap benchmark |
| `gold` | COMEX Gold Futures | `GC=F` | Yahoo Finance | Liquid USD gold proxy; widely used in Indian academic studies |
| `bond` | 13-Week T-Bill Yield | `^IRX` | Yahoo Finance | Short-rate proxy; replace with `0P00009OSF.BO` (SBI Magnum Gilt) for India-specific runs |
| `vix` | CBOE VIX (optional) | `^VIX` | Yahoo Finance | Volatility regime indicator; swap to `^INDIAVIX` when available |

All ticker defaults are defined in [`src/regime_shift/config.py`](../src/regime_shift/config.py)  
(`TickerConfig`) and can be overridden at runtime without touching any other module.

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
