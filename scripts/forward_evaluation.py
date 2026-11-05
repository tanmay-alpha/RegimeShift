"""Freeze once, then append evaluations of local user-supplied CSV batches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from regime_shift.research.forward import evaluate_forward, freeze_experiment  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Create a new immutable local freeze before evaluation")
    freeze.add_argument("--experiment-dir", required=True, type=Path)
    freeze.add_argument("--dataset", required=True, type=Path, help="Historical local CSV ending exactly at cutoff")
    freeze.add_argument("--cutoff", required=True, help="Last historical trading date, YYYY-MM-DD")
    freeze.add_argument("--config", type=Path, help="Complete RegimeShiftConfig JSON, not a partial override")
    freeze.add_argument("--release-at", help="Optional declared release timestamp with timezone; not independently verified")
    freeze.add_argument("--observed-at", help="Optional declared observation timestamp with timezone; defaults to local UTC clock")
    evaluate = commands.add_parser("evaluate", help="Append one new local CSV batch; never overwrite an attempt")
    evaluate.add_argument("--experiment-dir", required=True, type=Path)
    evaluate.add_argument("--dataset", required=True, type=Path, help="Only new rows, strictly after cutoff and prior accepted end")
    evaluate.add_argument("--observed-at", help="Declared observation timestamp with timezone; defaults to local UTC clock")
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_experiment(args.experiment_dir, dataset_path=args.dataset,
                                       cutoff=args.cutoff, config=args.config,
                                       observed_at=args.observed_at, release_at=args.release_at)
            output = {key: result[key] for key in ("experiment_id", "freeze_sha256", "observed_at", "cutoff")}
        else:
            result = evaluate_forward(args.experiment_dir, dataset_path=args.dataset,
                                      observed_at=args.observed_at)
            output = {key: result[key] for key in ("attempt_id", "freeze_sha256", "evaluation_type", "observed_at")}
        output["prospective_certified"] = False
        print(json.dumps(output, indent=2))
        return 0
    except Exception as exc:
        print(f"Forward protocol rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
