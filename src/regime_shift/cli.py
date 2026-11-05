"""UI-independent offline research CLI: ``python -m regime_shift.cli``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from regime_shift.research.workflow import (
    DEFAULT_RESULTS,
    WorkflowError,
    export_experiment_zip,
    inspect_experiment,
    list_experiments,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RegimeShift offline research lab; no live trading or data downloads.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data", help="Explicit ex-post historical data repair; never point-in-time")
    prepare.add_argument("--input", type=Path, default=Path("data/submission_market_data.csv"))
    prepare.add_argument("--output", type=Path, default=Path("data/cache/validated_market_data.csv"))
    run = commands.add_parser("run", help="Run one explicit JSON-configured experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    run.add_argument("--registry-dir", type=Path, default=Path("results/registry"))
    inspect = commands.add_parser("inspect", help="Inspect a manifest, or list existing results without running models")
    inspect.add_argument("experiment", type=Path, nargs="?")
    inspect.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    export = commands.add_parser("export", help="Export generated reproducibility artifacts, excluding raw prices")
    export.add_argument("experiment", type=Path)
    export.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare-data":
            from regime_shift.data import load_market_data_csv
            from regime_shift.experiment import sha256_file

            if args.output.exists() or args.output.with_suffix(".provenance.json").exists():
                raise ValueError("Prepared dataset output already exists; choose a new path to preserve evidence.")
            prices = load_market_data_csv(str(args.input))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            prices.to_csv(args.output, index_label="Date", float_format="%.12g")
            provenance = {
                "source_path": args.input.as_posix(), "source_sha256": sha256_file(args.input),
                "prepared_sha256": sha256_file(args.output),
                "policy": "Retrospective spike-and-reversal repair and bounded fill; NOT point-in-time.",
                "diagnostics": prices.attrs.get("data_diagnostics", {}),
            }
            args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            print(json.dumps(provenance, indent=2))
        elif args.command == "run":
            result = run_experiment(args.config, results_dir=args.results_dir, registry_dir=args.registry_dir)
            print(json.dumps({"status": "completed", "experiment_dir": str(result.output_dir),
                              "summary": result.manifest["summary"]}, indent=2))
        elif args.command == "inspect":
            if args.experiment is not None:
                print(json.dumps(inspect_experiment(args.experiment), indent=2))
            else:
                rows = [{"experiment_dir": str(path), "status": inspect_experiment(path).get("status"),
                         "summary": inspect_experiment(path).get("summary")} for path in list_experiments(args.results_dir)]
                print(json.dumps(rows, indent=2))
        elif args.command == "export":
            export_experiment_zip(args.experiment, args.output)
            print(str(args.output))
    except WorkflowError as exc:
        print(json.dumps({"status": "failed", "error": str(exc), "experiment_dir": str(exc.output_dir)}), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
