# Research protocol

RegimeShift tests whether a three-state, train-only HMM can improve allocation risk control. It does not assume alpha and it does not tune to preserve a historical Sharpe.

The full 2010–2026 dataset was viewed before this protocol. Development (2010–2018) and validation (2019–2021) are retrospective research splits; 2022–2026 is explicitly a retrospective evaluation period, not a pristine holdout. The only genuine lockbox is data later than the committed dataset end date, evaluated with `config/frozen_resume_v1.yaml` unchanged.

The official model is `NEXT_CLOSE`: existing weights earn close(t-1) to close(t); after close(t), all inputs through t may be fitted, a target is traded, costs are charged, and that target first earns close(t) to close(t+1). HMM restarts are selected by train-window likelihood only. Costs are asset-level stated assumptions (5/10/20 bps one-way scenarios), not claimed observations. Market impact and capacity are excluded because no executable volume/ADV data is supplied.

`^NSEI` is an index series and the current output is therefore an index-level allocation simulation, not proof of executable ETF fills. Gold and defensive series are ETF/NAV proxies. A future tradable experiment requires independently verified common-history adjusted prices, liquidity and corporate-action treatment for every instrument.
