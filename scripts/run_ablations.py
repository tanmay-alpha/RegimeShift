"""Generate controlled causal policy-ablation artefacts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from regime_shift.research.ablations import run_ablations

if __name__ == "__main__":
    run_ablations(Path(__file__).resolve().parents[1] / "results" / "research")
