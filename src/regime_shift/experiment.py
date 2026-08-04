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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def write_manifest(output_dir: str | Path, *, dataset_path: str, config: Mapping[str, Any], metadata: Mapping[str, Any]) -> Path:
    """Write a result manifest after outputs exist, including their hashes."""
    destination = Path(output_dir)
    result_hashes = {
        path.name: sha256_file(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "experiment_manifest.json"
    }
    encoded_config = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    manifest = {
        "experiment_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{git_commit()[:8]}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "configuration": config,
        "configuration_sha256": hashlib.sha256(encoded_config).hexdigest(),
        "software": {"python": sys.version.split()[0], "platform": platform.platform()},
        "metadata": dict(metadata),
        "result_file_sha256": result_hashes,
    }
    path = destination / "experiment_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
