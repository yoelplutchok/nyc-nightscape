"""
Quality assurance utilities.

- CRS mismatches are hard errors
- Bounds sanity checks on every geometry read/write
- FDR correction for multiple testing
- Valid CD set for filtering
"""

from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from nightscape.io_utils import read_yaml
from nightscape.paths import CONFIG_DIR


_BOUNDS_CONFIG_CACHE = None


def _load_bounds_config() -> dict:
    """Load bounds check configuration from params.yml (cached after first call)."""
    global _BOUNDS_CONFIG_CACHE
    if _BOUNDS_CONFIG_CACHE is None:
        params = read_yaml(CONFIG_DIR / "params.yml")
        _BOUNDS_CONFIG_CACHE = params.get("bounds_checks", {})
    return _BOUNDS_CONFIG_CACHE


class CRSError(Exception):
    pass


class BoundsError(Exception):
    pass


def assert_crs_not_none(gdf: gpd.GeoDataFrame, context: str = "") -> None:
    if gdf.crs is None:
        msg = "GeoDataFrame has no CRS set"
        if context:
            msg = f"{msg} ({context})"
        raise CRSError(msg)


def assert_expected_crs(gdf: gpd.GeoDataFrame, expected_epsg: int, context: str = "") -> None:
    assert_crs_not_none(gdf, context)
    expected_crs = CRS.from_epsg(expected_epsg)
    if not gdf.crs.equals(expected_crs):
        msg = f"CRS mismatch: expected EPSG:{expected_epsg}, got {gdf.crs}"
        if context:
            msg = f"{msg} ({context})"
        raise CRSError(msg)


def safe_reproject(gdf: gpd.GeoDataFrame, target_epsg: int, context: str = "") -> gpd.GeoDataFrame:
    """Safely reproject a GeoDataFrame to target CRS."""
    assert_crs_not_none(gdf, context)
    target_crs = CRS.from_epsg(target_epsg)
    if gdf.crs.equals(target_crs):
        return gdf
    return gdf.to_crs(target_crs)


def check_bounds_epsg4326(
    gdf: gpd.GeoDataFrame,
    lon_min: float = -75.0, lon_max: float = -73.0,
    lat_min: float = 40.0, lat_max: float = 41.5,
    context: str = "",
) -> bool:
    minx, miny, maxx, maxy = tuple(gdf.total_bounds)
    errors = []
    if minx < lon_min or maxx > lon_max:
        errors.append(f"Longitude out of range: [{minx}, {maxx}]")
    if miny < lat_min or maxy > lat_max:
        errors.append(f"Latitude out of range: [{miny}, {maxy}]")
    if errors:
        msg = "EPSG:4326 bounds check failed: " + "; ".join(errors)
        if context:
            msg = f"{msg} ({context})"
        raise BoundsError(msg)
    return True


def check_bounds_epsg2263(
    gdf: gpd.GeoDataFrame,
    x_min: float = 900000, x_max: float = 1100000,
    y_min: float = 110000, y_max: float = 280000,
    context: str = "",
) -> bool:
    minx, miny, maxx, maxy = tuple(gdf.total_bounds)
    errors = []
    if minx < x_min or maxx > x_max:
        errors.append(f"X out of range: [{minx}, {maxx}]")
    if miny < y_min or maxy > y_max:
        errors.append(f"Y out of range: [{miny}, {maxy}]")
    if errors:
        msg = "EPSG:2263 bounds check failed: " + "; ".join(errors)
        if context:
            msg = f"{msg} ({context})"
        raise BoundsError(msg)
    return True


def validate_bounds(gdf: gpd.GeoDataFrame, context: str = "") -> bool:
    """Validate bounds based on the GeoDataFrame's CRS."""
    assert_crs_not_none(gdf, context)
    epsg = gdf.crs.to_epsg()
    bounds_config = _load_bounds_config()

    if epsg == 4326:
        config = bounds_config.get("epsg_4326", {})
        return check_bounds_epsg4326(gdf, **config, context=context) if config else check_bounds_epsg4326(gdf, context=context)
    elif epsg == 2263:
        config = bounds_config.get("epsg_2263", {})
        return check_bounds_epsg2263(gdf, **config, context=context) if config else check_bounds_epsg2263(gdf, context=context)
    else:
        bounds = tuple(gdf.total_bounds)
        if not all(np.isfinite(bounds)):
            raise BoundsError(f"Non-finite bounds: {bounds} ({context})")
        return True


# =============================================================================
# FDR Correction
# =============================================================================

def apply_fdr_correction(
    p_values: pd.Series,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Apply FDR correction to a series of p-values."""
    from statsmodels.stats.multitest import multipletests

    mask = p_values.notna()
    result = pd.DataFrame({
        "p_value": p_values,
        "p_adjusted": np.nan,
        "significant_fdr": False,
    }, index=p_values.index)

    if mask.sum() > 0:
        reject, p_adj, _, _ = multipletests(
            p_values[mask].values, alpha=alpha, method=method)
        result.loc[mask, "p_adjusted"] = p_adj
        result.loc[mask, "significant_fdr"] = reject

    return result


def fdr_correct_analysis_df(
    df: pd.DataFrame,
    p_col: str = "p_value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add FDR-corrected p-values to an analysis results DataFrame."""
    if p_col not in df.columns:
        return df

    correction = apply_fdr_correction(df[p_col], alpha=alpha)
    result = df.copy()
    result["p_adjusted"] = correction["p_adjusted"].values
    result["significant_fdr"] = correction["significant_fdr"].values
    return result


# =============================================================================
# Valid NYC Community Districts
# =============================================================================

def get_valid_cds() -> set:
    """Return the set of 59 valid NYC community district boro_cd codes."""
    valid = set()
    for boro, max_cd in [(1, 12), (2, 12), (3, 18), (4, 14), (5, 3)]:
        for cd in range(1, max_cd + 1):
            valid.add(boro * 100 + cd)
    return valid


VALID_CDS = get_valid_cds()


def filter_standard_cds(df: pd.DataFrame, cd_col: str = "boro_cd") -> pd.DataFrame:
    """Filter a DataFrame to only standard 59 NYC community districts."""
    if cd_col not in df.columns:
        return df
    return df[df[cd_col].isin(VALID_CDS)].copy()
