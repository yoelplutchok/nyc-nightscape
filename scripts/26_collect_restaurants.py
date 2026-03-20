"""
26_collect_restaurants.py — Fetch restaurant density by CD from DOHMH inspections.

Dataset ID: 43nn-pn8j (DOHMH Restaurant Inspections)
Has lat/lon for spatial join to CDs. Uses unique CAMIS (restaurant ID) for counting.
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

DATASET_ID = "43nn-pn8j"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "restaurants"
OUTPUT_PATH = REPORTS_DIR / "restaurant_density_cd.csv"

# Late-night food types (proxy for places likely open late)
LATE_NIGHT_CUISINES = [
    "Pizza", "Pizza/Italian", "Donuts", "Bagels/Pretzels",
    "Hamburgers", "Chicken", "Chinese", "Mexican",
    "Latin American", "Caribbean", "Korean", "Japanese",
    "Delicatessen", "Sandwiches", "Hotdogs",
]


def fetch_restaurants(logger) -> pd.DataFrame:
    """Fetch unique restaurants with lat/lon."""
    cache_path = CACHE_DIR / "restaurants_raw.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    # Get unique restaurants (distinct CAMIS) with their most recent inspection
    # Use server-side to get distinct restaurants with location
    all_rows = []
    offset = 0

    while True:
        api_params = {
            "$select": "camis, dba, boro, cuisine_description, latitude, longitude",
            "$where": "latitude > 0",
            "$limit": 50000,
            "$offset": offset,
        }

        logger.info(f"Fetching restaurants page at offset {offset}...")
        resp = requests.get(BASE_URL, params=api_params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)
        logger.info(f"  Got {len(data)} rows (total: {len(all_rows)})")

        if len(data) < 50000:
            break
        offset += 50000

    df = pd.DataFrame(all_rows)
    # Deduplicate by CAMIS (unique restaurant identifier) since $group was removed
    if "camis" in df.columns:
        df = df.drop_duplicates(subset=["camis"], keep="first")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached {len(df)} unique restaurants")

    return df


def process(df: pd.DataFrame, params: dict, logger) -> pd.DataFrame:
    """Spatial join and compute density metrics."""
    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"] > 0]

    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    cd = gpd.read_file(GEO_DIR / "cd59.geojson")
    cd["boro_cd"] = cd["boro_cd"].astype(int)
    cd = cd[cd["boro_cd"].isin(VALID_CDS)][["boro_cd", "geometry"]]

    joined = gpd.sjoin(gdf, cd, how="inner", predicate="within")
    logger.info(f"Spatial join: {len(df)} restaurants -> {len(joined)} matched")

    jdf = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))

    # Total restaurants per CD
    total = jdf.groupby("boro_cd")["camis"].nunique().reset_index()
    total.columns = ["boro_cd", "restaurant_count"]

    # Late-night food types
    late_night = jdf[jdf["cuisine_description"].isin(LATE_NIGHT_CUISINES)]
    ln_agg = late_night.groupby("boro_cd")["camis"].nunique().reset_index()
    ln_agg.columns = ["boro_cd", "late_night_food_count"]

    # Cuisine diversity (number of unique cuisine types)
    diversity = jdf.groupby("boro_cd")["cuisine_description"].nunique().reset_index()
    diversity.columns = ["boro_cd", "cuisine_diversity"]

    result = total.merge(ln_agg, on="boro_cd", how="left").merge(diversity, on="boro_cd", how="left")
    result = result.fillna(0)

    # Population rates
    master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    result = result.merge(pop, on="boro_cd", how="left")

    result["restaurants_per_1k"] = (
        result["restaurant_count"] / (result["population"] / 1000)
    ).replace([float("inf"), float("-inf")], 0).fillna(0)
    result["late_night_food_per_1k"] = (
        result["late_night_food_count"] / (result["population"] / 1000)
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    logger.log_metrics({
        "total_restaurants": int(result["restaurant_count"].sum()),
        "late_night_food_count": int(result["late_night_food_count"].sum()),
        "mean_cuisine_diversity": result["cuisine_diversity"].mean(),
    })

    return result


def main():
    with get_logger("26_collect_restaurants") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        raw_df = fetch_restaurants(logger)
        result = process(raw_df, params, logger)

        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population")]
        result[fill_cols] = result[fill_cols].fillna(0)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)

        print(f"\nRestaurant density:")
        print(f"  Total restaurants: {int(result['restaurant_count'].sum()):,}")
        print(f"  Late-night food types: {int(result['late_night_food_count'].sum()):,}")
        print(f"  Mean restaurants/1K: {result['restaurants_per_1k'].mean():.1f}")
        print(f"  Mean cuisine diversity: {result['cuisine_diversity'].mean():.0f} types")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
