# RegimeShift data contract

`submission_market_data.csv` is the canonical offline dataset for the frozen research run. It contains 4,090 source rows from 2010-01-04 through 2026-07-27. The causal common evaluation horizon begins 2010-10-05 after feature and training warm-up.

| Column | Series | Role and tradability |
|---|---|---|
| `equity` | NIFTY 50 Index (`^NSEI`) | Index-level return and signal series; not directly executable. |
| `gold` | Nippon India ETF Gold Bees (`GOLDBEES.NS`) | INR-denominated gold ETF proxy. |
| `bond` | Nippon India Liquid Bees (`LIQUIDBEES.NS`) | Short-duration liquid-bond/cash-equivalent proxy; not a sovereign bond, gilt fund, or 10-year government security. |

`^INDIAVIX` is an optional feature-only series and is omitted from the official dataset. No gilt-fund series is part of the active universe.

The canonical SHA-256 is `290c783af9fb803334388090c3b72df7a31f5fd580de62bb7a905250777226e7`. CSV hashing normalizes CRLF and LF to LF before hashing, so manifests remain portable across operating systems.

The committed data is included for academic reproducibility only. It remains subject to the relevant market-data provider, exchange, and instrument terms; the MIT license does not license third-party market data. See [`../DATA_LICENSE_NOTICE.md`](../DATA_LICENSE_NOTICE.md).
