"""
22_collect_citibike.py — Fetch late-night Citi Bike trips (2021-2023).

Source: Citi Bike System Data — annual ZIPs on S3
URL: https://s3.amazonaws.com/tripdata/YYYY-citibike-tripdata.zip
Each ZIP contains monthly CSVs. ~1.5 GB per year.

Filter to nighttime start hours (22:00-06:59), spatial join to CDs.
"""

import io
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

CACHE_DIR = RAW_DIR / "citibike"
OUTPUT_PATH = REPORTS_DIR / "late_night_citibike_cd.csv"


def download_year(year: int, logger) -> Path:
    """Download annual Citi Bike zip."""
    url = f"https://s3.amazonaws.com/tripdata/{year}-citibike-tripdata.zip"
    zip_path = CACHE_DIR / f"{year}-citibike-tripdata.zip"

    if zip_path.exists():
        logger.info(f"  Cached: {zip_path.name}")
        return zip_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading {year} Citi Bike data (~1.5 GB)...")
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    logger.info(f"  Downloaded: {zip_path.stat().st_size / 1e9:.2f} GB")
    return zip_path


def _extract_csvs(zip_path: Path, logger):
    """Yield (csv_name, file_obj) from annual zip, handling nested zips."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Check for direct CSVs first
        csv_names = sorted([n for n in zf.namelist() if n.endswith(".csv")])
        seen_csvs = set()
        if csv_names:
            for csv_name in csv_names:
                seen_csvs.add(csv_name)
                yield csv_name, zf.open(csv_name)

        # Nested zips (annual zip → monthly zips → CSVs)
        inner_zips = sorted([
            n for n in zf.namelist()
            if n.endswith(".zip") and "__MACOSX" not in n
        ])
        if inner_zips:
            logger.info(f"  {len(inner_zips)} monthly zips inside")
        for iz_name in inner_zips:
            with zf.open(iz_name) as iz_f:
                inner_data = io.BytesIO(iz_f.read())
                try:
                    with zipfile.ZipFile(inner_data) as izf:
                        inner_csvs = sorted([n for n in izf.namelist() if n.endswith(".csv")])
                        for csv_name in inner_csvs:
                            if csv_name not in seen_csvs:
                                seen_csvs.add(csv_name)
                                yield csv_name, izf.open(csv_name)
                except zipfile.BadZipFile:
                    logger.warning(f"  Skipping bad inner zip: {iz_name}")


def process_year(zip_path: Path, logger) -> pd.DataFrame:
    """Extract nighttime trips from annual zip (process CSV by CSV)."""
    night_trips = []

    csv_count = 0
    for csv_name, csv_file in _extract_csvs(zip_path, logger):
        csv_count += 1
        logger.info(f"    Processing {csv_name}...")
        with csv_file as f:
            df = pd.read_csv(f, low_memory=False)

            # Find datetime column
            time_col = None
            for col in ["started_at", "starttime", "start_time"]:
                if col in df.columns:
                    time_col = col
                    break
            if time_col is None:
                continue

            df["started_at_parsed"] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=["started_at_parsed"])
            df["hour"] = df["started_at_parsed"].dt.hour
            night = df[(df["hour"] >= 22) | (df["hour"] < 7)]

            # Get coordinates
            lat_col = "start_lat" if "start_lat" in night.columns else "start station latitude"
            lon_col = "start_lng" if "start_lng" in night.columns else "start station longitude"

            if lat_col not in night.columns or lon_col not in night.columns:
                continue

            subset = night[[lat_col, lon_col]].copy()
            subset.columns = ["lat", "lng"]
            subset["lat"] = pd.to_numeric(subset["lat"], errors="coerce")
            subset["lng"] = pd.to_numeric(subset["lng"], errors="coerce")
            subset = subset.dropna()
            subset = subset[
                (subset["lat"] > 40) & (subset["lat"] < 42) &
                (subset["lng"] > -75) & (subset["lng"] < -73)
            ]

            night_trips.append(subset)
            logger.info(f"      {len(subset)} nighttime trips")

    logger.info(f"  Processed {csv_count} CSVs total")
    if night_trips:
        return pd.concat(night_trips, ignore_index=True)
    return pd.DataFrame()


def main():
    with get_logger("22_collect_citibike") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        year_start = params["time_windows"]["primary"]["year_start"]
        year_end = params["time_windows"]["primary"]["year_end"]

        # Check for pre-aggregated cache
        agg_cache = CACHE_DIR / "citibike_night_trips_agg.parquet"
        if agg_cache.exists():
            logger.info(f"Loading pre-aggregated cache from {agg_cache}")
            all_trips = pd.read_parquet(agg_cache)
        else:
            all_years = []
            for year in range(year_start, year_end + 1):
                logger.info(f"Processing {year}...")
                zip_path = download_year(year, logger)
                trips = process_year(zip_path, logger)
                if not trips.empty:
                    all_years.append(trips)
                    logger.info(f"  Year {year}: {len(trips)} nighttime trips")

            if not all_years:
                logger.error("No Citi Bike data collected")
                return

            all_trips = pd.concat(all_years, ignore_index=True)
            all_trips.to_parquet(agg_cache, index=False)
            logger.info(f"Total nighttime trips: {len(all_trips)}")

        # Spatial join to CDs
        geometry = [Point(lon, lat) for lon, lat in zip(all_trips["lng"], all_trips["lat"])]
        gdf = gpd.GeoDataFrame(all_trips, geometry=geometry, crs="EPSG:4326")

        cd = gpd.read_file(GEO_DIR / "cd59.geojson")
        cd["boro_cd"] = cd["boro_cd"].astype(int)
        cd = cd[cd["boro_cd"].isin(VALID_CDS)][["boro_cd", "geometry"]]

        joined = gpd.sjoin(gdf, cd, how="inner", predicate="within")
        logger.info(f"Spatial join: {len(all_trips)} -> {len(joined)} matched")

        jdf = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))

        # Aggregate
        agg = jdf.groupby("boro_cd").size().reset_index(name="citibike_trips_night")

        # Approximate station count per CD
        jdf["station_loc"] = jdf["lat"].round(4).astype(str) + "," + jdf["lng"].round(4).astype(str)
        stations = jdf.groupby("boro_cd")["station_loc"].nunique().reset_index()
        stations.columns = ["boro_cd", "citibike_stations"]

        result = agg.merge(stations, on="boro_cd", how="left")

        # Rates
        years = year_end - year_start + 1
        master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
        pop = master[["boro_cd", "population"]].copy()
        pop["boro_cd"] = pop["boro_cd"].astype(int)
        result = result.merge(pop, on="boro_cd", how="left")

        cd_proj = cd.to_crs("EPSG:2263")
        cd_areas = cd.copy()
        cd_areas["area_km2"] = cd_proj.geometry.area * 9.2903e-8
        result = result.merge(cd_areas[["boro_cd", "area_km2"]], on="boro_cd", how="left")

        result["citibike_trips_night_per_1k"] = (
            result["citibike_trips_night"] / years
            / (result["population"] / 1000)
        ).replace([float("inf"), float("-inf")], 0).fillna(0)

        result["citibike_stations_per_km2"] = (
            result["citibike_stations"] / result["area_km2"]
        ).replace([float("inf"), float("-inf")], 0).fillna(0)

        # Ensure all 59 CDs
        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population", "area_km2")]
        result[fill_cols] = result[fill_cols].fillna(0)

        result["citibike_coverage_flag"] = (result["citibike_stations"] > 0).astype(int)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)

        print(f"\nCiti Bike late-night trips (2021-2023):")
        print(f"  Total nighttime trips: {int(result['citibike_trips_night'].sum()):,}")
        print(f"  CDs with coverage: {int(result['citibike_coverage_flag'].sum())}/59")
        print(f"  Mean trips/1K/yr: {result['citibike_trips_night_per_1k'].mean():.0f}")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
