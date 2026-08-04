"""Generate rolling-origin artefacts from the causal ablation series."""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from regime_shift.research.rolling_origin import run_rolling_origin

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]; out = root / "results" / "research"
    run_rolling_origin(pd.read_csv(out / "ablation_daily_returns.csv", index_col="date", parse_dates=True), out)
