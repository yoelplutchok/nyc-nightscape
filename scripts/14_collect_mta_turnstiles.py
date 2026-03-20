"""
14_collect_mta_turnstiles.py — Fetch late-night subway ridership (2021-2023).

Source: MTA Subway Hourly Ridership (data.ny.gov, wujg-7c2s)
Uses server-side aggregation by station with lat/lon, then spatial joins to CDs.
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

DATASET_ID = "wujg-7c2s"
BASE_URL = f"https://data.ny.gov/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "mta_turnstiles"
OUTPUT_PATH = REPORTS_DIR / "late_night_subway_cd.csv"


def fetch_station_ridership(params: dict, logger, time_filter: str, label: str) -> pd.DataFrame:
    """Fetch ridership aggregated by station for a given time filter."""
    year_start = params["time_windows"]["primary"]["year_start"]
    year_end = params["time_windows"]["primary"]["year_end"]

    where_clause = (
        f"transit_mode='subway' "
        f"AND transit_timestamp >= '{year_start}-01-01T00:00:00' "
        f"AND transit_timestamp < '{year_end + 1}-01-01T00:00:00' "
        f"AND {time_filter}"
    )

    select_clause = (
        "station_complex_id, station_complex, borough, latitude, longitude, "
        "sum(ridership) as total_ridership"
    )

    api_params = {
        "$where": where_clause,
        "$select": select_clause,
        "$group": "station_complex_id, station_complex, borough, latitude, longitude",
        "$limit": 50000,
    }

    logger.info(f"Fetching {label} subway ridership...")
    resp = requests.get(BASE_URL, params=api_params, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    logger.info(f"  Got {len(data)} stations for {label}")
    return pd.DataFrame(data)


def main():
    with get_logger("14_collect_mta_turnstiles") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        cache_night = CACHE_DIR / "subway_late_night_raw.parquet"
        cache_all = CACHE_DIR / "subway_all_hours_raw.parquet"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Late-night ridership (22:00-06:59)
        if cache_night.exists():
            logger.info("Loading cached late-night data")
            night_df = pd.read_parquet(cache_night)
        else:
            night_filter = "(date_extract_hh(transit_timestamp) >= 22 OR date_extract_hh(transit_timestamp) < 7)"
            night_df = fetch_station_ridership(params, logger, night_filter, "late-night")
            night_df.to_parquet(cache_night, index=False)

        # All-hours ridership (for late-night share)
        if cache_all.exists():
            logger.info("Loading cached all-hours data")
            all_df = pd.read_parquet(cache_all)
        else:
            all_df = fetch_station_ridership(params, logger, "1=1", "all-hours")
            all_df.to_parquet(cache_all, index=False)

        # Clean numeric fields
        for df in [night_df, all_df]:
            df["total_ridership"] = pd.to_numeric(df["total_ridership"], errors="coerce").fillna(0)
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        # Spatial join to CDs
        cd = gpd.read_file(GEO_DIR / "cd59.geojson")
        cd["boro_cd"] = cd["boro_cd"].astype(int)
        cd = cd[cd["boro_cd"].isin(VALID_CDS)][["boro_cd", "geometry"]]

        def spatial_join_stations(df, label):
            df = df.dropna(subset=["latitude", "longitude"]).copy()
            geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            joined = gpd.sjoin(gdf, cd, how="inner", predicate="within")
            logger.info(f"  {label}: {len(df)} stations -> {len(joined)} matched to CDs")
            return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))

        night_joined = spatial_join_stations(night_df, "late-night")
        all_joined = spatial_join_stations(all_df, "all-hours")

        # Aggregate by CD
        night_agg = night_joined.groupby("boro_cd").agg(
            late_night_ridership=("total_ridership", "sum"),
            subway_stations=("station_complex_id", "nunique"),
        ).reset_index()

        all_agg = all_joined.groupby("boro_cd")["total_ridership"].sum().reset_index()
        all_agg.columns = ["boro_cd", "total_ridership"]

        result = night_agg.merge(all_agg, on="boro_cd", how="left")
        result["pct_ridership_late_night"] = (
            result["late_night_ridership"] / result["total_ridership"] * 100
        ).replace([float("inf"), float("-inf")], 0).fillna(0)

        # Population and area rates
        master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
        pop = master[["boro_cd", "population"]].copy()
        pop["boro_cd"] = pop["boro_cd"].astype(int)
        result = result.merge(pop, on="boro_cd", how="left")

        # Area for station density
        cd_proj = cd.to_crs("EPSG:2263")
        cd_areas = cd.copy()
        cd_areas["area_km2"] = cd_proj.geometry.area * 9.2903e-8  # sq feet to km2
        result = result.merge(cd_areas[["boro_cd", "area_km2"]], on="boro_cd", how="left")

        years = params["time_windows"]["primary"]["year_end"] - params["time_windows"]["primary"]["year_start"] + 1

        result["late_night_entries_per_1k"] = (
            result["late_night_ridership"] / years
            / (result["population"] / 1000)
        ).replace([float("inf"), float("-inf")], 0).fillna(0)

        result["subway_stations_per_km2"] = (
            result["subway_stations"] / result["area_km2"]
        ).replace([float("inf"), float("-inf")], 0).fillna(0)

        # Ensure all 59 CDs
        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population", "area_km2")]
        result[fill_cols] = result[fill_cols].fillna(0)

        # Transit desert flag: bottom quartile of late-night ridership per capita
        # Computed AFTER all 59 CDs are present so zero-ridership CDs are included
        threshold = result["late_night_entries_per_1k"].quantile(0.25)
        result["transit_desert_flag"] = (result["late_night_entries_per_1k"] <= threshold).astype(int)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)
        logger.info(f"Saved {len(result)} rows to {OUTPUT_PATH}")

        logger.log_metrics({
            "total_late_night_ridership": int(result["late_night_ridership"].sum()),
            "total_stations_matched": int(result["subway_stations"].sum()),
            "transit_deserts": int(result["transit_desert_flag"].sum()),
        })

        print(f"\nLate-night subway ridership (2021-2023):")
        print(f"  Total late-night ridership: {int(result['late_night_ridership'].sum()):,.0f}")
        print(f"  Stations matched to CDs: {int(result['subway_stations'].sum())}")
        print(f"  CDs with stations: {(result['subway_stations'] > 0).sum()}/59")
        print(f"  Transit desert CDs: {int(result['transit_desert_flag'].sum())}")
        print(f"  Mean late-night entries/1K/yr: {result['late_night_entries_per_1k'].mean():.0f}")
        print(f"  Mean % late-night: {result['pct_ridership_late_night'].mean():.1f}%")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
