"""
Hashing utilities for reproducibility and cache validation.

Each output has a metadata sidecar with input file hashes,
config digest, code version, library versions, and timestamp.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nightscape.io_utils import atomic_write_json, read_json
from nightscape.logging_utils import get_versions
from nightscape.paths import METADATA_DIR


def hash_file(path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")

    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_string(s: str, algorithm: str = "sha256") -> str:
    """Compute hash of a string."""
    h = hashlib.new(algorithm)
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def hash_dict(d: Dict[str, Any], algorithm: str = "sha256") -> str:
    """Compute hash of a dictionary via deterministic JSON serialization."""
    s = json.dumps(d, sort_keys=True, default=str)
    return hash_string(s, algorithm)


def _get_project_root() -> Path:
    """Get project root for git commands."""
    try:
        from nightscape.paths import PROJECT_ROOT
        return PROJECT_ROOT
    except ImportError:
        return Path(__file__).resolve().parent.parent.parent


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=_get_project_root(),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_git_dirty() -> Optional[bool]:
    """Check if git working directory has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
            cwd=_get_project_root(),
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_git_info() -> Dict[str, Any]:
    """Get git repository information."""
    return {
        "commit": get_git_commit(),
        "dirty": get_git_dirty(),
    }


def write_metadata_sidecar(
    output_path: Union[str, Path],
    inputs: Dict[str, str],
    config: Dict[str, Any],
    run_id: str,
    extra: Optional[Dict[str, Any]] = None,
    metadata_dir: Optional[Path] = None,
) -> Path:
    """Write a metadata sidecar file for an output."""
    if metadata_dir is None:
        metadata_dir = METADATA_DIR

    output_path = Path(output_path)

    input_hashes = {}
    for name, path in inputs.items():
        path = Path(path)
        if path.exists():
            input_hashes[name] = {"path": str(path), "hash": hash_file(path)}
        else:
            input_hashes[name] = {"path": str(path), "hash": None, "missing": True}

    metadata = {
        "output_file": str(output_path),
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": input_hashes,
        "config_digest": hash_dict(config),
        "config": config,
        "git": get_git_info(),
        "versions": get_versions(),
    }
    if extra:
        metadata["extra"] = extra

    suffix_tag = output_path.suffix.lstrip(".") if output_path.suffix else "file"
    sidecar_name = f"{output_path.stem}_{suffix_tag}_metadata.json"
    sidecar_path = metadata_dir / sidecar_name
    atomic_write_json(metadata, sidecar_path)
    return sidecar_path


def read_metadata_sidecar(output_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read metadata sidecar for an output file."""
    output_path = Path(output_path)
    suffix_tag = output_path.suffix.lstrip(".") if output_path.suffix else "file"
    sidecar_name = f"{output_path.stem}_{suffix_tag}_metadata.json"
    sidecar_path = METADATA_DIR / sidecar_name
    if sidecar_path.exists():
        return read_json(sidecar_path)
    return None


def validate_cache(
    output_path: Union[str, Path],
    inputs: Dict[str, str],
    config: Dict[str, Any],
) -> bool:
    """Check if cached output is still valid based on input hashes and config."""
    output_path = Path(output_path)

    if not output_path.exists():
        return False

    metadata = read_metadata_sidecar(output_path)
    if metadata is None:
        return False

    if metadata.get("config_digest") != hash_dict(config):
        return False

    cached_inputs = metadata.get("inputs", {})
    for name, path in inputs.items():
        path = Path(path)
        if name not in cached_inputs:
            return False
        if not path.exists():
            return False
        if cached_inputs[name].get("hash") != hash_file(path):
            return False

    return True
