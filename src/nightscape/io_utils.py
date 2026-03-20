"""
I/O utilities with atomic writes and safe reads.

All outputs written via temp file -> rename/replace for atomicity.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import geopandas as gpd
import pandas as pd
import yaml


# =============================================================================
# Atomic Write Utilities
# =============================================================================

@contextmanager
def atomic_write(
    target_path: Union[str, Path],
    mode: str = "w",
    suffix: Optional[str] = None,
):
    """Context manager for atomic file writes via temp file + rename."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix is None:
        suffix = target_path.suffix or ".tmp"

    fd, temp_path = tempfile.mkstemp(
        suffix=suffix,
        prefix=f".{target_path.stem}_",
        dir=target_path.parent,
    )
    temp_path = Path(temp_path)

    try:
        os.close(fd)
        with open(temp_path, mode) as f:
            yield f
        temp_path.replace(target_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_df(
    df: pd.DataFrame,
    target_path: Union[str, Path],
    **kwargs,
) -> None:
    """Atomically write a DataFrame to CSV or Parquet."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = target_path.suffix.lower()
    fd, temp_path = tempfile.mkstemp(
        suffix=suffix,
        prefix=f".{target_path.stem}_",
        dir=target_path.parent,
    )
    os.close(fd)
    temp_path = Path(temp_path)

    try:
        if suffix == ".parquet":
            df.to_parquet(temp_path, **kwargs)
        elif suffix == ".csv":
            df.to_csv(temp_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
        temp_path.replace(target_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_gdf(
    gdf: gpd.GeoDataFrame,
    target_path: Union[str, Path],
    **kwargs,
) -> None:
    """Atomically write a GeoDataFrame to GeoParquet or GeoJSON."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = target_path.suffix.lower()
    fd, temp_path = tempfile.mkstemp(
        suffix=suffix,
        prefix=f".{target_path.stem}_",
        dir=target_path.parent,
    )
    os.close(fd)
    temp_path = Path(temp_path)

    try:
        if suffix == ".parquet":
            gdf.to_parquet(temp_path, **kwargs)
        elif suffix == ".geojson":
            gdf.to_file(temp_path, driver="GeoJSON", **kwargs)
        elif suffix == ".gpkg":
            gdf.to_file(temp_path, driver="GPKG", **kwargs)
        else:
            raise ValueError(f"Unsupported geo format: {suffix}")
        temp_path.replace(target_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_json(
    data: Any,
    target_path: Union[str, Path],
    **kwargs,
) -> None:
    """Atomically write JSON data."""
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("default", str)

    with atomic_write(target_path, mode="w", suffix=".json") as f:
        json.dump(data, f, **kwargs)


def atomic_write_yaml(
    data: Any,
    target_path: Union[str, Path],
    **kwargs,
) -> None:
    """Atomically write YAML data."""
    kwargs.setdefault("default_flow_style", False)
    kwargs.setdefault("sort_keys", False)

    with atomic_write(target_path, mode="w", suffix=".yml") as f:
        yaml.safe_dump(data, f, **kwargs)


# =============================================================================
# Read Utilities
# =============================================================================

def read_yaml(path: Union[str, Path]) -> dict:
    """Read a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_json(path: Union[str, Path]) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_gdf(path: Union[str, Path], **kwargs) -> gpd.GeoDataFrame:
    """Read a GeoDataFrame from file (GeoParquet, GeoJSON, etc.)."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return gpd.read_parquet(path, **kwargs)
    else:
        return gpd.read_file(path, **kwargs)


def read_df(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read a DataFrame from CSV or Parquet."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    elif suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {suffix}")
