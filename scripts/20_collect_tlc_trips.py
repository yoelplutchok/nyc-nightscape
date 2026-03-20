"""
20_collect_tlc_trips.py — Fetch nighttime TLC taxi/rideshare trips (2021-2023).

Source: TLC Trip Records (Parquet files on NYC TLC S3)
Yellow taxi: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet
Green taxi: https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_YYYY-MM.parquet
FHVHV (Uber/Lyft): https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_YYYY-MM.parquet

Strategy: Sample every 4th month (Jan, May, Sep) to keep download manageable.
Filter to nighttime pickup hours (22:00-06:59).
Spatial join pickup lat/lon to CDs.
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

CDN_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
CACHE_DIR = RAW_DIR / "tlc_trips"
OUTPUT_PATH = REPORTS_DIR / "nighttime_tlc_trips_cd.csv"

# Sample months: Jan, May, Sep of each year
SAMPLE_MONTHS = [1, 5, 9]


def download_parquet(url: str, local_path: Path, logger) -> bool:
    """Download a Parquet file if not cached."""
    if local_path.exists():
        logger.info(f"  Cached: {local_path.name}")
        return True

    logger.info(f"  Downloading {local_path.name}...")
    try:
        resp = requests.get(url, timeout=300, stream=True)
        if resp.status_code != 200:
            logger.warning(f"  HTTP {resp.status_code} for {url}")
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"  Downloaded: {local_path.name} ({local_path.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        logger.warning(f"  Failed: {e}")
        return False


def process_yellow(path: Path, logger) -> pd.DataFrame:
    """Extract nighttime pickups from yellow taxi data."""
    df = pd.read_parquet(path, columns=[
        "tpep_pickup_datetime", "PULocationID",
    ])
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    night = df[(df["hour"] >= 22) | (df["hour"] < 7)]
    agg = night.groupby("PULocationID").size().reset_index(name="yellow_trips")
    return agg


def process_green(path: Path, logger) -> pd.DataFrame:
    """Extract nighttime pickups from green taxi data."""
    df = pd.read_parquet(path, columns=[
        "lpep_pickup_datetime", "PULocationID",
    ])
    df["hour"] = df["lpep_pickup_datetime"].dt.hour
    night = df[(df["hour"] >= 22) | (df["hour"] < 7)]
    agg = night.groupby("PULocationID").size().reset_index(name="green_trips")
    return agg


def process_fhvhv(path: Path, logger) -> pd.DataFrame:
    """Extract nighttime pickups from FHVHV (Uber/Lyft) data."""
    df = pd.read_parquet(path, columns=[
        "pickup_datetime", "PULocationID",
    ])
    df["hour"] = df["pickup_datetime"].dt.hour
    night = df[(df["hour"] >= 22) | (df["hour"] < 7)]
    agg = night.groupby("PULocationID").size().reset_index(name="fhvhv_trips")
    return agg


def load_taxi_zone_to_cd(logger) -> pd.DataFrame:
    """Build TLC taxi zone -> community district crosswalk via spatial join."""
    cache_xwalk = CACHE_DIR / "taxi_zone_cd_crosswalk.csv"
    if cache_xwalk.exists():
        return pd.read_csv(cache_xwalk)

    # Download taxi zone shapefile
    zone_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    zone_path = CACHE_DIR / "taxi_zones.zip"

    if not zone_path.exists():
        logger.info("Downloading taxi zone shapefile...")
        resp = requests.get(zone_url, timeout=120)
        resp.raise_for_status()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(zone_path, "wb") as f:
            f.write(resp.content)

    import zipfile
    extract_dir = CACHE_DIR / "taxi_zones"
    if not extract_dir.exists():
        with zipfile.ZipFile(zone_path, "r") as zf:
            zf.extractall(extract_dir)

    # Find the shapefile inside
    shp_files = list(extract_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found in {extract_dir}")
    zones = gpd.read_file(shp_files[0])
    zones = zones.to_crs("EPSG:4326")

    cd = gpd.read_file(GEO_DIR / "cd59.geojson")
    cd["boro_cd"] = cd["boro_cd"].astype(int)
    cd = cd[cd["boro_cd"].isin(VALID_CDS)]

    # Spatial join zone centroids to CDs
    zones["centroid"] = zones.geometry.centroid
    zone_points = zones.set_geometry("centroid")[["LocationID", "centroid"]].rename(columns={"centroid": "geometry"})
    zone_points = gpd.GeoDataFrame(zone_points, geometry="geometry", crs="EPSG:4326")

    joined = gpd.sjoin(zone_points, cd[["boro_cd", "geometry"]], how="left", predicate="within")
    xwalk = joined[["LocationID", "boro_cd"]].dropna().copy()
    xwalk["LocationID"] = xwalk["LocationID"].astype(int)
    xwalk["boro_cd"] = xwalk["boro_cd"].astype(int)

    xwalk.to_csv(cache_xwalk, index=False)
    logger.info(f"Built taxi zone -> CD crosswalk: {len(xwalk)} zones mapped")

    return xwalk


def main():
    with get_logger("20_collect_tlc_trips") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        year_start = params["time_windows"]["primary"]["year_start"]
        year_end = params["time_windows"]["primary"]["year_end"]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Build crosswalk
        xwalk = load_taxi_zone_to_cd(logger)

        all_yellow = []
        all_green = []
        all_fhvhv = []

        for year in range(year_start, year_end + 1):
            for month in SAMPLE_MONTHS:
                ym = f"{year}-{month:02d}"
                logger.info(f"Processing {ym}...")

                # Yellow
                ypath = CACHE_DIR / f"yellow_tripdata_{ym}.parquet"
                yurl = f"{CDN_BASE}/yellow_tripdata_{ym}.parquet"
                if download_parquet(yurl, ypath, logger):
                    try:
                        all_yellow.append(process_yellow(ypath, logger))
                    except Exception as e:
                        logger.warning(f"  Yellow {ym} failed: {e}")

                # Green
                gpath = CACHE_DIR / f"green_tripdata_{ym}.parquet"
                gurl = f"{CDN_BASE}/green_tripdata_{ym}.parquet"
                if download_parquet(gurl, gpath, logger):
                    try:
                        all_green.append(process_green(gpath, logger))
                    except Exception as e:
                        logger.warning(f"  Green {ym} failed: {e}")

                # FHVHV
                fpath = CACHE_DIR / f"fhvhv_tripdata_{ym}.parquet"
                furl = f"{CDN_BASE}/fhvhv_tripdata_{ym}.parquet"
                if download_parquet(furl, fpath, logger):
                    try:
                        all_fhvhv.append(process_fhvhv(fpath, logger))
                    except Exception as e:
                        logger.warning(f"  FHVHV {ym} failed: {e}")

        # Combine across months
        def combine_agg(frames, trip_col, loc_col="PULocationID"):
            if not frames:
                return pd.DataFrame({loc_col: [], trip_col: []})
            combined = pd.concat(frames)
            return combined.groupby(loc_col)[trip_col].sum().reset_index()

        yellow_agg = combine_agg(all_yellow, "yellow_trips")
        green_agg = combine_agg(all_green, "green_trips")
        fhvhv_agg = combine_agg(all_fhvhv, "fhvhv_trips")

        # Merge all trip types
        merged = yellow_agg.merge(green_agg, on="PULocationID", how="outer") \
                           .merge(fhvhv_agg, on="PULocationID", how="outer")
        merged = merged.fillna(0)

        # Map to CDs via crosswalk
        merged = merged.merge(xwalk, left_on="PULocationID", right_on="LocationID", how="inner")

        cd_agg = merged.groupby("boro_cd").agg(
            yellow_trips=("yellow_trips", "sum"),
            green_trips=("green_trips", "sum"),
            fhvhv_trips=("fhvhv_trips", "sum"),
        ).reset_index()

        # Scale from sampled months to full year estimate
        # We sampled 3 months per year x 3 years = 9 months out of 36
        scale_factor = 12 / len(SAMPLE_MONTHS)  # 4x to annualize from 3 sampled months/year
        years = year_end - year_start + 1

        cd_agg["taxi_trips_night"] = cd_agg["yellow_trips"] + cd_agg["green_trips"]
        cd_agg["rideshare_trips_night"] = cd_agg["fhvhv_trips"]
        cd_agg["total_ride_trips_night"] = cd_agg["taxi_trips_night"] + cd_agg["rideshare_trips_night"]

        # Population rates (annualized, scaled)
        master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
        pop = master[["boro_cd", "population"]].copy()
        pop["boro_cd"] = pop["boro_cd"].astype(int)
        cd_agg = cd_agg.merge(pop, on="boro_cd", how="left")

        pop_per_1k = cd_agg["population"] / 1000
        cd_agg["taxi_pickups_night_per_1k"] = (
            cd_agg["taxi_trips_night"] * scale_factor / years / pop_per_1k
        ).replace([float("inf"), float("-inf")], 0).fillna(0)

        cd_agg["rideshare_pickups_night_per_1k"] = (
            cd_agg["rideshare_trips_night"] * scale_factor / years / pop_per_1k
        ).replace([float("inf"), float("-inf")], 0).fillna(0)

        cd_agg["total_ride_pickups_night_per_1k"] = (
            cd_agg["total_ride_trips_night"] * scale_factor / years / pop_per_1k
        ).replace([float("inf"), float("-inf")], 0).fillna(0)

        cd_agg["rideshare_to_taxi_ratio"] = (
            cd_agg["rideshare_trips_night"] / cd_agg["taxi_trips_night"]
        ).fillna(0).replace([float("inf"), float("-inf")], 0)

        # Ensure all 59 CDs
        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(cd_agg, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population")]
        result[fill_cols] = result[fill_cols].fillna(0)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)

        logger.log_metrics({
            "sampled_months": len(SAMPLE_MONTHS) * years,
            "yellow_trips_sampled": int(cd_agg["yellow_trips"].sum()),
            "green_trips_sampled": int(cd_agg["green_trips"].sum()),
            "fhvhv_trips_sampled": int(cd_agg["fhvhv_trips"].sum()),
        })

        print(f"\nTLC nighttime trips (sampled {len(SAMPLE_MONTHS)} months/yr, 2021-2023):")
        print(f"  Yellow taxi (sampled): {int(result['yellow_trips'].sum()):,}")
        print(f"  Green taxi (sampled): {int(result['green_trips'].sum()):,}")
        print(f"  FHVHV/rideshare (sampled): {int(result['fhvhv_trips'].sum()):,}")
        print(f"  Mean total ride pickups/1K/yr: {result['total_ride_pickups_night_per_1k'].mean():.0f}")
        print(f"  Mean rideshare:taxi ratio: {result['rideshare_to_taxi_ratio'].mean():.1f}")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
