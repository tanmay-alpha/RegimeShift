"""Run the frozen, no-tuning research suite and write public artefacts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from regime_shift.research.ablations import run_ablations
from regime_shift.research.bootstrap import run_bootstrap
from regime_shift.research.hmm_diagnostics import run_hmm_diagnostics
from regime_shift.research.rolling_origin import run_rolling_origin
from regime_shift.research.subperiods import run_subperiods

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]; out = root / "results" / "research"
    daily = run_ablations(out)
    run_subperiods(daily, out)
    run_rolling_origin(daily, out)
    run_bootstrap(daily, out)
    run_hmm_diagnostics(out)
    print(f"Research suite complete: {out}")
