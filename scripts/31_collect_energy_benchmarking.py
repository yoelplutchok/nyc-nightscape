"""
31_collect_energy_benchmarking.py — Building energy benchmarking by CD.

Dataset ID: 28fi-3us3 (NYC Benchmarking Law - Energy/Water)
Fallback IDs (public, year-specific LL84 disclosures):
    7x5e-2fxh  (2022, calendar year 2021)
    usc3-8zwd  (2021, calendar year 2020)

Buildings report site EUI, electricity use, and GFA.  We aggregate to
community districts using the ``community_board`` field (which stores
boro_cd directly, e.g. "401"), with a spatial-join fallback for rows
missing that field.

Output variables:
    energy_use_per_sqft     — mean site EUI (kBtu/ft²) per CD
    electricity_use_per_sqft — mean electricity intensity per CD
    large_buildings_per_km2 — count of benchmarked buildings / CD area
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import atomic_write_df, read_yaml, read_df
from nightscape.logging_utils import get_logger
from nightscape.paths import RAW_DIR, REPORTS_DIR, CONFIG_DIR, GEO_DIR
from nightscape.qa import VALID_CDS, filter_standard_cds
from nightscape.schemas import ensure_boro_cd_dtype

DATASET_ID = "28fi-3us3"
# Public year-specific fallbacks (most recent first)
FALLBACK_IDS = ["7x5e-2fxh", "usc3-8zwd"]
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "energy_benchmarking"
OUTPUT_PATH = REPORTS_DIR / "energy_benchmarking_cd.csv"
PAGE_SIZE = 50000

# Socrata field names (from the LL84 2022 schema)
COL_SITE_EUI = "site_eui_kbtu_ft"
COL_ELEC_KBTU = "electricity_use_grid_purchase"
COL_GFA = "property_gfa_self_reported"
COL_GFA_CALC = "property_gfa_calculated"
COL_LAT = "latitude"
COL_LON = "longitude"
COL_CB = "community_board"
COL_PROPERTY_ID = "property_id"
COL_PROPERTY_NAME = "property_name"


# ------------------------------------------------------------------
# Fetch
# ------------------------------------------------------------------

def _try_fetch(dataset_id: str, logger) -> list:
    """Attempt paginated fetch from a single Socrata dataset.

    Returns a list of dicts on success, or an empty list on 403/404.
    """
    url = f"https://data.cityofnewyork.us/resource/{dataset_id}.json"
    all_rows: list = []
    offset = 0

    while True:
        api_params = {
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }
        logger.info(f"Fetching dataset {dataset_id} page at offset {offset}...")
        try:
            resp = requests.get(url, params=api_params, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            if resp.status_code in (403, 404):
                logger.warning(
                    f"Dataset {dataset_id} returned {resp.status_code} — "
                    "may require authentication or is unavailable."
                )
                return []
            raise exc

        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            logger.warning(f"Dataset {dataset_id} error: {data.get('message')}")
            return []

        if not data:
            break

        all_rows.extend(data)
        logger.info(f"  Got {len(data)} rows (total: {len(all_rows)})")

        if len(data) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    return all_rows


def fetch_energy_data(logger) -> pd.DataFrame:
    """Fetch building energy benchmarking data, with fallback datasets."""
    cache_path = CACHE_DIR / "energy_benchmarking_raw.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    # Try primary dataset first, then fallbacks
    datasets_to_try = [DATASET_ID] + FALLBACK_IDS
    all_rows: list = []

    for ds_id in datasets_to_try:
        rows = _try_fetch(ds_id, logger)
        if rows:
            logger.info(f"Successfully fetched {len(rows)} rows from {ds_id}")
            all_rows = rows
            break
        logger.info(f"No data from {ds_id}, trying next fallback...")

    if not all_rows:
        logger.error("Could not fetch energy benchmarking data from any source.")
        raise RuntimeError(
            "All energy benchmarking dataset sources failed. "
            "Check network or NYC Open Data status."
        )

    df = pd.DataFrame(all_rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached {len(df)} rows to {cache_path}")

    return df


# ------------------------------------------------------------------
# Process
# ------------------------------------------------------------------

def _assign_cd_from_community_board(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Parse the ``community_board`` field into integer boro_cd.

    The field stores values like "401" (Queens CD 1), "101" (Manhattan CD 1).
    Returns the dataframe with a new ``boro_cd`` column (NaN where unparseable).
    """
    df = df.copy()

    if COL_CB not in df.columns:
        logger.warning(f"Column '{COL_CB}' not found — will rely on spatial join")
        df["boro_cd"] = pd.NA
        return df

    df["boro_cd"] = pd.to_numeric(df[COL_CB], errors="coerce").astype("Int64")
    # Keep only valid CDs
    df.loc[~df["boro_cd"].isin(VALID_CDS), "boro_cd"] = pd.NA

    n_direct = df["boro_cd"].notna().sum()
    logger.info(f"Direct CD assignment via community_board: {n_direct}/{len(df)} rows")
    return df


def _spatial_join_remaining(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Spatial-join rows that still lack a boro_cd using lat/lon."""
    missing_mask = df["boro_cd"].isna()
    n_missing = missing_mask.sum()
    if n_missing == 0:
        return df

    has_coords = (
        missing_mask
        & df[COL_LAT].notna()
        & df[COL_LON].notna()
        & (pd.to_numeric(df[COL_LAT], errors="coerce") > 0)
    )
    n_with_coords = has_coords.sum()
    logger.info(f"Spatial join for {n_with_coords} rows missing CD but having lat/lon")

    if n_with_coords == 0:
        return df

    subset = df.loc[has_coords].copy()
    subset[COL_LAT] = pd.to_numeric(subset[COL_LAT], errors="coerce")
    subset[COL_LON] = pd.to_numeric(subset[COL_LON], errors="coerce")

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(subset[COL_LON], subset[COL_LAT])
    ]
    gdf = gpd.GeoDataFrame(subset, geometry=geometry, crs="EPSG:4326")

    cd = gpd.read_file(GEO_DIR / "cd59.geojson")
    cd["boro_cd"] = cd["boro_cd"].astype(int)
    cd = cd[cd["boro_cd"].isin(VALID_CDS)][["boro_cd", "geometry"]]

    joined = gpd.sjoin(gdf, cd, how="left", predicate="within")
    # Deduplicate points that matched multiple polygons on boundaries
    joined = joined[~joined.index.duplicated(keep="first")]
    # sjoin gives boro_cd_left (original NA) and boro_cd_right (matched)
    if "boro_cd_right" in joined.columns:
        matched_cds = joined["boro_cd_right"]
    elif "index_right" in joined.columns:
        # boro_cd came through directly from the right frame
        matched_cds = joined["boro_cd"]
    else:
        matched_cds = pd.Series(pd.NA, index=joined.index)

    df.loc[has_coords, "boro_cd"] = matched_cds.values
    n_recovered = df.loc[has_coords, "boro_cd"].notna().sum()
    logger.info(f"Spatial join recovered CD for {n_recovered}/{n_with_coords} rows")

    return df


def process(df: pd.DataFrame, params: dict, logger) -> pd.DataFrame:
    """Aggregate building energy data to community districts."""
    df = df.copy()

    # --- Parse numeric energy columns ---
    for col in [COL_SITE_EUI, COL_ELEC_KBTU, COL_GFA, COL_GFA_CALC, COL_LAT, COL_LON]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use self-reported GFA, fall back to calculated
    if COL_GFA in df.columns and COL_GFA_CALC in df.columns:
        df["gfa"] = df[COL_GFA].fillna(df[COL_GFA_CALC])
    elif COL_GFA in df.columns:
        df["gfa"] = df[COL_GFA]
    elif COL_GFA_CALC in df.columns:
        df["gfa"] = df[COL_GFA_CALC]
    else:
        df["gfa"] = pd.NA

    # --- Assign community district ---
    df = _assign_cd_from_community_board(df, logger)
    df = _spatial_join_remaining(df, logger)

    # Drop rows with no CD assignment
    n_before = len(df)
    df = df.dropna(subset=["boro_cd"])
    df["boro_cd"] = df["boro_cd"].astype(int)
    logger.info(f"Rows with valid CD: {len(df)}/{n_before}")

    # --- Compute electricity intensity (kBtu / ft²) ---
    if COL_ELEC_KBTU in df.columns:
        df["elec_intensity"] = df[COL_ELEC_KBTU] / df["gfa"]
        df.loc[~df["elec_intensity"].between(0, 10000), "elec_intensity"] = pd.NA
    else:
        df["elec_intensity"] = pd.NA

    # Sanity-clip site EUI (drop extreme outliers)
    if COL_SITE_EUI in df.columns:
        df.loc[~df[COL_SITE_EUI].between(0, 10000), COL_SITE_EUI] = pd.NA

    # --- Aggregate by CD ---
    agg_dict = {}
    if COL_SITE_EUI in df.columns:
        agg_dict["energy_use_per_sqft"] = (COL_SITE_EUI, "mean")
    if "elec_intensity" in df.columns:
        agg_dict["electricity_use_per_sqft"] = ("elec_intensity", "mean")
    agg_dict["building_count"] = (COL_SITE_EUI if COL_SITE_EUI in df.columns else "boro_cd", "count")

    grouped = df.groupby("boro_cd").agg(**agg_dict).reset_index()

    # --- Building density per km² ---
    cd_geo = gpd.read_file(GEO_DIR / "cd59.geojson")
    cd_geo["boro_cd"] = cd_geo["boro_cd"].astype(int)
    cd_proj = cd_geo.to_crs("EPSG:2263")
    # EPSG:2263 is in US survey feet; 1 sq foot = 9.2903e-8 km²
    cd_geo["area_km2"] = cd_proj.geometry.area * 9.2903e-8
    areas = cd_geo[["boro_cd", "area_km2"]]

    grouped = grouped.merge(areas, on="boro_cd", how="left")
    grouped["large_buildings_per_km2"] = (
        grouped["building_count"] / grouped["area_km2"]
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    # --- Population merge ---
    master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    grouped = grouped.merge(pop, on="boro_cd", how="left")

    logger.log_metrics({
        "total_buildings": int(grouped["building_count"].sum()),
        "cds_with_data": int((grouped["building_count"] > 0).sum()),
        "mean_site_eui": float(grouped["energy_use_per_sqft"].mean())
        if "energy_use_per_sqft" in grouped.columns else None,
        "mean_elec_intensity": float(grouped["electricity_use_per_sqft"].mean())
        if "electricity_use_per_sqft" in grouped.columns else None,
        "mean_buildings_per_km2": float(grouped["large_buildings_per_km2"].mean()),
    })

    return grouped


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    with get_logger("31_collect_energy_benchmarking") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        raw_df = fetch_energy_data(logger)
        result = process(raw_df, params, logger)

        # Ensure all 59 CDs present
        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [
            c for c in result.columns
            if c not in ("boro_cd", "population", "area_km2")
        ]
        result[fill_cols] = result[fill_cols].fillna(0)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)

        print(f"\nEnergy benchmarking by community district:")
        print(f"  Total benchmarked buildings: {int(result['building_count'].sum()):,}")
        print(f"  CDs with data: {(result['building_count'] > 0).sum()}/59")
        if "energy_use_per_sqft" in result.columns:
            active = result[result["building_count"] > 0]
            print(f"  Mean site EUI (kBtu/ft²): {active['energy_use_per_sqft'].mean():.1f}")
        if "electricity_use_per_sqft" in result.columns:
            active = result[result["building_count"] > 0]
            print(f"  Mean elec intensity (kBtu/ft²): {active['electricity_use_per_sqft'].mean():.1f}")
        print(f"  Mean buildings/km²: {result['large_buildings_per_km2'].mean():.1f}")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
