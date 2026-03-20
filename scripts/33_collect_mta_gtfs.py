"""
33_collect_mta_gtfs.py — Subway late-night service frequency from GTFS.

Source: MTA GTFS Static Feed
URL: http://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip

Computes scheduled train frequency during late-night hours (midnight-6AM)
per station, then spatial joins stations to CDs.
"""

import sys
import zipfile
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

GTFS_URL = "http://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"
CACHE_DIR = RAW_DIR / "mta_gtfs"
OUTPUT_PATH = REPORTS_DIR / "subway_frequency_cd.csv"


def download_gtfs(logger) -> Path:
    """Download GTFS static feed."""
    zip_path = CACHE_DIR / "gtfs_subway.zip"
    if zip_path.exists():
        logger.info(f"Cached: {zip_path.name}")
        return zip_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MTA GTFS subway feed...")
    resp = requests.get(GTFS_URL, timeout=120)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(resp.content)
    logger.info(f"Downloaded: {zip_path.stat().st_size / 1e6:.1f} MB")
    return zip_path


def load_gtfs_tables(zip_path: Path) -> dict:
    """Load key GTFS tables from zip."""
    tables = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in ["stops.txt", "stop_times.txt", "trips.txt",
                      "routes.txt", "calendar.txt"]:
            if name in zf.namelist():
                with zf.open(name) as f:
                    tables[name.replace(".txt", "")] = pd.read_csv(f, low_memory=False)
    return tables


def process(tables: dict, logger, params: dict = None) -> pd.DataFrame:
    """Compute late-night train frequency per station, join to CDs."""
    stops = tables["stops"]
    stop_times = tables["stop_times"]
    trips = tables["trips"]

    # Read late-night window from config (default: 1-5 per params.yml)
    if params is not None:
        ln_start = params["time_windows"]["late_night"]["start_hour"]
        ln_end = params["time_windows"]["late_night"]["end_hour"]
    else:
        ln_start, ln_end = 1, 5
    ln_hours = ln_end - ln_start  # number of hours in window

    # Parse stop times to get hour
    stop_times = stop_times.copy()
    # GTFS times can exceed 24:00 (e.g., 25:30 = 1:30 AM next day)
    def parse_hour(t):
        try:
            h = int(str(t).split(":")[0])
            return h % 24
        except (ValueError, IndexError):
            return -1

    stop_times["hour"] = stop_times["departure_time"].apply(parse_hour)

    # Late night: use config window (default 01:00-04:59)
    late_night_mask = (
        (stop_times["hour"] >= ln_start) & (stop_times["hour"] < ln_end)
    )
    late_night = stop_times[late_night_mask].copy()

    # Count trips per stop during late night
    # Join with trips to get route info
    late_night = late_night.merge(
        trips[["trip_id", "route_id", "service_id"]],
        on="trip_id", how="left"
    )

    # Get weekday service IDs (most representative)
    weekday_services = []
    if "calendar" in tables:
        cal = tables["calendar"]
        weekday_services = cal[
            (cal["monday"] == 1) | (cal["tuesday"] == 1) |
            (cal["wednesday"] == 1) | (cal["thursday"] == 1) |
            (cal["friday"] == 1)
        ]["service_id"].unique()
        if len(weekday_services) > 0:
            late_night = late_night[late_night["service_id"].isin(weekday_services)]

    # Parent stations: stop_id ending in N/S are direction variants
    # Use parent_station if available, otherwise strip N/S suffix
    if "parent_station" in stops.columns:
        stop_parent = stops[["stop_id", "parent_station"]].copy()
        stop_parent["station_id"] = stop_parent["parent_station"].fillna(stop_parent["stop_id"])
    else:
        stops["station_id"] = stops["stop_id"].str.rstrip("NS")
        stop_parent = stops[["stop_id", "station_id"]]

    late_night = late_night.merge(stop_parent, on="stop_id", how="left")

    # Unique trips per station per hour (proxy for frequency)
    station_trips = late_night.groupby("station_id").agg(
        late_night_trips=("trip_id", "nunique"),
        late_night_routes=("route_id", "nunique"),
    ).reset_index()

    # Trains per hour (use config-derived window length)
    station_trips["trains_per_hour_late_night"] = station_trips["late_night_trips"] / ln_hours

    # Get station locations (parent stations only)
    station_locs = stops.copy()
    if "parent_station" in station_locs.columns:
        # Use parent station rows (location_type == 1) or stops without parents
        parents = station_locs[
            (station_locs["location_type"] == 1) |
            (station_locs["parent_station"].isna()) |
            (station_locs["parent_station"] == "")
        ].copy()
        parents["station_id"] = parents["stop_id"]
    else:
        parents = station_locs.copy()
        parents["station_id"] = parents["stop_id"].str.rstrip("NS")

    parents = parents.drop_duplicates(subset=["station_id"])
    parents["stop_lat"] = pd.to_numeric(parents["stop_lat"], errors="coerce")
    parents["stop_lon"] = pd.to_numeric(parents["stop_lon"], errors="coerce")
    parents = parents.dropna(subset=["stop_lat", "stop_lon"])

    station_trips = station_trips.merge(
        parents[["station_id", "stop_lat", "stop_lon", "stop_name"]],
        on="station_id", how="inner"
    )

    logger.info(f"Stations with late-night service: {len(station_trips)}")

    # Spatial join to CDs
    geometry = [
        Point(lon, lat) for lon, lat in
        zip(station_trips["stop_lon"], station_trips["stop_lat"])
    ]
    gdf = gpd.GeoDataFrame(station_trips, geometry=geometry, crs="EPSG:4326")

    cd = gpd.read_file(GEO_DIR / "cd59.geojson")
    cd["boro_cd"] = cd["boro_cd"].astype(int)
    cd = cd[cd["boro_cd"].isin(VALID_CDS)][["boro_cd", "geometry"]]

    joined = gpd.sjoin(gdf, cd, how="inner", predicate="within")
    logger.info(f"Spatial join: {len(station_trips)} stations -> {len(joined)} matched")

    # Aggregate per CD
    cd_agg = joined.groupby("boro_cd").agg(
        subway_stations_late_night=("station_id", "nunique"),
        subway_trips_late_night=("late_night_trips", "sum"),
        subway_routes_late_night=("route_id", "nunique"),
        mean_trains_per_hour_late_night=("trains_per_hour_late_night", "mean"),
    ).reset_index()

    # Also find last train hour per station (approximate from stop_times)
    # For simplicity, use the max hour in 0-5 range
    all_hours = stop_times.merge(stop_parent, on="stop_id", how="left")
    if "calendar" in tables and len(weekday_services) > 0:
        all_hours = all_hours.merge(trips[["trip_id", "service_id"]], on="trip_id", how="left")
        all_hours = all_hours[all_hours["service_id"].isin(weekday_services)]

    # Get the latest departure hour for each station
    max_hour = all_hours.groupby("station_id")["hour"].max().reset_index()
    max_hour.columns = ["station_id", "last_service_hour"]

    max_hour = max_hour.merge(
        parents[["station_id", "stop_lat", "stop_lon"]],
        on="station_id", how="inner"
    )
    max_hour_geo = gpd.GeoDataFrame(
        max_hour,
        geometry=[Point(lon, lat) for lon, lat in zip(max_hour["stop_lon"], max_hour["stop_lat"])],
        crs="EPSG:4326"
    )
    max_hour_joined = gpd.sjoin(max_hour_geo, cd, how="inner", predicate="within")
    last_train = max_hour_joined.groupby("boro_cd")["last_service_hour"].max().reset_index()
    last_train.columns = ["boro_cd", "subway_last_train_hour"]

    cd_agg = cd_agg.merge(last_train, on="boro_cd", how="left")

    # Population for per-capita rates
    master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    cd_agg = cd_agg.merge(pop, on="boro_cd", how="left")

    cd_agg["subway_late_night_trips_per_1k"] = (
        cd_agg["subway_trips_late_night"] / (cd_agg["population"] / 1000)
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    # Ensure all 59 CDs
    all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
    result = all_cds.merge(cd_agg, on="boro_cd", how="left")
    fill_cols = [c for c in result.columns if c not in ("boro_cd", "population")]
    result[fill_cols] = result[fill_cols].fillna(0)

    # Transit service gap: flag CDs with 0 late-night stations
    result["late_night_transit_gap"] = (result["subway_stations_late_night"] == 0).astype(int)

    logger.log_metrics({
        "stations_with_late_night": int((result["subway_stations_late_night"] > 0).sum()),
        "mean_trains_per_hour": result["mean_trains_per_hour_late_night"].mean(),
        "transit_gap_cds": int(result["late_night_transit_gap"].sum()),
    })

    return result


def main():
    with get_logger("33_collect_mta_gtfs") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        zip_path = download_gtfs(logger)
        tables = load_gtfs_tables(zip_path)
        logger.info(f"GTFS tables loaded: {list(tables.keys())}")

        result = process(tables, logger, params)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)

        print(f"\nSubway Late-Night Frequency (GTFS):")
        print(f"  CDs with late-night service: {int((result['subway_stations_late_night'] > 0).sum())}/59")
        print(f"  Mean trains/hr (midnight-6AM): {result['mean_trains_per_hour_late_night'].mean():.1f}")
        print(f"  Transit gap CDs: {int(result['late_night_transit_gap'].sum())}")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
