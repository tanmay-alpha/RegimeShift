"""Machine-readable manifests for reproducible RegimeShift experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    """Hash a dataset identically on Windows and Unix checkouts.

    CSV source files are text artefacts and Git may normalize their line endings.
    Canonicalizing only CSV line endings prevents an otherwise identical dataset
    from invalidating a release manifest on CI.
    """
    digest = hashlib.sha256()
    target = Path(path)
    with target.open("rb") as handle:
        if target.suffix.lower() == ".csv":
            digest.update(handle.read().replace(b"\r\n", b"\n"))
            return digest.hexdigest()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def source_tree_dirty() -> bool:
    """Check source inputs only, intentionally excluding generated artefacts."""
    paths = ["src", "tests", "scripts", "config", "run_submission.py", "pyproject.toml", "requirements.txt"]
    result = subprocess.run(["git", "status", "--porcelain", "--", *paths], capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())


def write_manifest(output_dir: str | Path, *, dataset_path: str, config: Mapping[str, Any], metadata: Mapping[str, Any]) -> Path:
    """Write a result manifest after outputs exist, including their hashes."""
    destination = Path(output_dir)
    result_hashes = {
        path.name: sha256_file(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "experiment_manifest.json"
    }
    encoded_config = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    root = Path.cwd()
    dataset = Path(dataset_path)
    try:
        relative_dataset = str(dataset.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        # Offline tests and ad-hoc research can supply external data.  Preserve
        # a portable logical identifier rather than leaking a local absolute path.
        relative_dataset = f"external/{dataset.name}"
    canonical_hash = sha256_file(dataset)
    manifest = {
        "experiment_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{git_commit()[:8]}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "generated_from_commit": git_commit(),
        "source_tree_dirty_before_generation": source_tree_dirty(),
        "ignored_generation_paths": ["results/submission", "results/research", "reports/build"],
        "dataset_path": relative_dataset,
        "dataset_sha256_canonical": canonical_hash,
        "dataset_hash_policy": "CRLF and LF normalized to LF before hashing",
        "configuration": config,
        "configuration_sha256": hashlib.sha256(encoded_config).hexdigest(),
        "software": {"python": sys.version.split()[0], "platform": platform.platform()},
        "metadata": dict(metadata),
        "result_file_sha256": result_hashes,
    }
    path = destination / "experiment_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
