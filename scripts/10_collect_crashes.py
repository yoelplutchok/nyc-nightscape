"""
10_collect_crashes.py — Fetch nighttime motor vehicle crashes (2021-2023) from NYC Open Data.

Dataset: NYPD Motor Vehicle Collisions - Crashes
Dataset ID: h9gi-nx95
API: https://data.cityofnewyork.us/resource/h9gi-nx95.json

Fetches point-level crash data with lat/lon, then spatial-joins to 59 community districts.
Outputs CD-level crash rates per 1K population.
"""

import json
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

DATASET_ID = "h9gi-nx95"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "crashes"
OUTPUT_PATH = REPORTS_DIR / "nighttime_crashes_cd.csv"
PAGE_SIZE = 50000


def fetch_nighttime_crashes(params: dict, logger) -> pd.DataFrame:
    """Fetch nighttime crash records with lat/lon via paginated API calls."""
    cache_path = CACHE_DIR / "nighttime_crashes_raw.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    year_start = params["time_windows"]["primary"]["year_start"]
    year_end = params["time_windows"]["primary"]["year_end"]

    # Fetch all crashes in date range (no server-side time filter) because
    # crash_time is a text field ("H:MM") that may lack leading zeros,
    # making SoQL string comparison unreliable for nighttime filtering.
    where_clause = (
        f"crash_date >= '{year_start}-01-01T00:00:00' "
        f"AND crash_date < '{year_end + 1}-01-01T00:00:00' "
        f"AND latitude IS NOT NULL"
    )

    select_fields = (
        "collision_id, crash_date, crash_time, borough, latitude, longitude, "
        "number_of_persons_injured, number_of_persons_killed, "
        "number_of_pedestrians_injured, number_of_pedestrians_killed, "
        "number_of_cyclist_injured, number_of_cyclist_killed, "
        "number_of_motorist_injured, number_of_motorist_killed"
    )

    all_rows = []
    offset = 0

    while True:
        api_params = {
            "$where": where_clause,
            "$select": select_fields,
            "$order": "collision_id",
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }

        logger.info(f"Fetching page at offset {offset}...")
        resp = requests.get(BASE_URL, params=api_params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)
        logger.info(f"  Got {len(data)} rows (total: {len(all_rows)})")

        if len(data) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    logger.info(f"Total crash records fetched: {len(all_rows)}")

    df = pd.DataFrame(all_rows)

    # Client-side nighttime filter: parse hour from crash_time ("H:MM" text)
    if "crash_time" in df.columns and len(df) > 0:
        df["_hour"] = df["crash_time"].str.split(":").str[0].astype(int)
        before = len(df)
        df = df[(df["_hour"] >= 22) | (df["_hour"] < 7)]
        df = df.drop(columns=["_hour"])
        logger.info(f"Nighttime filter: {before} -> {len(df)} rows (hours 22:00-06:59)")

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached {len(df)} rows to {cache_path}")

    return df


def spatial_join_to_cds(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Spatial join crash points to community district polygons."""
    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    cd = gpd.read_file(GEO_DIR / "cd59.geojson")
    if "boro_cd" not in cd.columns:
        # Try common column names
        for col in cd.columns:
            if "boro" in col.lower() and "cd" in col.lower():
                cd = cd.rename(columns={col: "boro_cd"})
                break

    cd["boro_cd"] = cd["boro_cd"].astype(int)
    cd = cd[cd["boro_cd"].isin(VALID_CDS)][["boro_cd", "geometry"]]

    joined = gpd.sjoin(gdf, cd, how="inner", predicate="within")
    logger.info(f"Spatial join: {len(df)} points -> {len(joined)} matched to CDs")
    drop_pct = (1 - len(joined) / len(df)) * 100
    logger.info(f"  Unmatched: {len(df) - len(joined)} ({drop_pct:.1f}%)")

    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


def compute_rates(df: pd.DataFrame, params: dict, logger) -> pd.DataFrame:
    """Aggregate crashes by CD and compute per-1K-population rates."""
    numeric_cols = [
        "number_of_persons_injured", "number_of_persons_killed",
        "number_of_pedestrians_injured", "number_of_pedestrians_killed",
        "number_of_cyclist_injured", "number_of_cyclist_killed",
        "number_of_motorist_injured", "number_of_motorist_killed",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    agg = df.groupby("boro_cd").agg(
        crash_count=("collision_id", "count"),
        persons_injured=("number_of_persons_injured", "sum"),
        persons_killed=("number_of_persons_killed", "sum"),
        ped_injured=("number_of_pedestrians_injured", "sum"),
        ped_killed=("number_of_pedestrians_killed", "sum"),
        cyclist_injured=("number_of_cyclist_injured", "sum"),
        cyclist_killed=("number_of_cyclist_killed", "sum"),
        motorist_injured=("number_of_motorist_injured", "sum"),
        motorist_killed=("number_of_motorist_killed", "sum"),
    ).reset_index()

    # Load population
    master_path = REPORTS_DIR / "master_analysis_df.parquet"
    master = read_df(master_path)
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    agg = agg.merge(pop, on="boro_cd", how="left")

    years = params["time_windows"]["primary"]["year_end"] - params["time_windows"]["primary"]["year_start"] + 1

    pop_per_1k = agg["population"] / 1000
    agg["crash_rate_per_1k"] = (
        (agg["crash_count"] / years) / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)
    agg["injury_rate_per_1k"] = (
        (agg["persons_injured"] / years) / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)
    agg["night_crash_fatality_rate"] = (
        (agg["persons_killed"] / years) / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)
    agg["night_ped_crash_rate"] = (
        (agg["ped_injured"] + agg["ped_killed"]) / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)
    agg["night_cyclist_crash_rate"] = (
        (agg["cyclist_injured"] + agg["cyclist_killed"]) / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)
    agg["night_crash_severity"] = (
        (agg["persons_injured"] + agg["persons_killed"] * 10) / agg["crash_count"]
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    logger.log_metrics({
        "total_nighttime_crashes": int(agg["crash_count"].sum()),
        "total_injuries": int(agg["persons_injured"].sum()),
        "total_fatalities": int(agg["persons_killed"].sum()),
        "cds_with_data": len(agg),
    })

    return agg


def main():
    with get_logger("10_collect_crashes") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        # Fetch
        raw_df = fetch_nighttime_crashes(params, logger)

        # Spatial join to CDs
        df = spatial_join_to_cds(raw_df, logger)

        # Compute rates
        result = compute_rates(df, params, logger)

        # Ensure all 59 CDs present
        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population")]
        result[fill_cols] = result[fill_cols].fillna(0)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        # Save
        atomic_write_df(result, OUTPUT_PATH, index=False)
        logger.info(f"Saved {len(result)} rows to {OUTPUT_PATH}")
        logger.log_outputs({"nighttime_crashes_cd": str(OUTPUT_PATH)})

        print(f"\nNighttime crashes (2021-2023):")
        print(f"  Total crashes: {int(result['crash_count'].sum()):,}")
        print(f"  Total injuries: {int(result['persons_injured'].sum()):,}")
        print(f"  Total fatalities: {int(result['persons_killed'].sum()):,}")
        print(f"  CDs with data: {(result['crash_count'] > 0).sum()}/59")
        print(f"  Mean crashes/1K/yr: {result['crash_rate_per_1k'].mean():.2f}")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
