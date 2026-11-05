"""Experiment registration: every attempt is recorded, including failures.

Convention for a versioned experiment directory ``results/experiments/<experiment_id>/``:

* ``manifest.json`` written by :func:`write_experiment_manifest` *after* all outputs
  exist: experiment id, protocol version, git commit, dirty flag, dataset hash,
  configuration, environment and SHA-256 of every output file.
* One line per attempt appended to ``results/registry.jsonl`` by
  :func:`register_attempt` (status ``completed`` or ``failed``, with the number of
  parameter trials evaluated, so selection bias can be accounted for later).
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from regime_shift.experiment import git_commit, sha256_file, source_tree_dirty
from regime_shift.research.spec import PROTOCOL_VERSION

REGISTRY_PATH = Path("results") / "registry.jsonl"


def environment() -> Dict[str, str]:
    import cvxpy
    import hmmlearn
    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "hmmlearn": hmmlearn.__version__, "cvxpy": cvxpy.__version__,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_experiment_manifest(output_dir: str | Path, *, experiment_id: str, description: str,
                              dataset_path: str | Path, config: Mapping[str, Any],
                              extra: Optional[Mapping[str, Any]] = None) -> Path:
    out = Path(output_dir)
    files = {p.relative_to(out).as_posix(): sha256_file(p)
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    manifest = {
        "experiment_id": experiment_id, "description": description, "protocol_version": PROTOCOL_VERSION,
        "created_utc": _utc_now(), "git_commit": git_commit(), "source_tree_dirty": source_tree_dirty(),
        "dataset_path": Path(dataset_path).as_posix(), "dataset_sha256_canonical": sha256_file(dataset_path),
        "config": dict(config), "environment": environment(), "outputs": files, **dict(extra or {}),
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def register_attempt(*, experiment_id: str, status: str, n_trials: int, summary: Mapping[str, Any],
                     dataset_path: str | Path, registry_path: str | Path = REGISTRY_PATH,
                     error: Optional[str] = None) -> Dict[str, Any]:
    """Append one attempt (never rewrite history). ``status`` is 'completed' or 'failed'."""
    if status not in ("completed", "failed"):
        raise ValueError("status must be 'completed' or 'failed'.")
    record = {
        "experiment_id": experiment_id, "status": status, "created_utc": _utc_now(),
        "protocol_version": PROTOCOL_VERSION, "git_commit": git_commit(),
        "source_tree_dirty": source_tree_dirty(), "dataset_sha256_canonical": sha256_file(dataset_path),
        "n_trials": int(n_trials), "summary": dict(summary), "error": error,
    }
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return record


def read_registry(registry_path: str | Path = REGISTRY_PATH) -> list[Dict[str, Any]]:
    path = Path(registry_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
