"""
21_collect_arrests.py — Fetch NYPD arrests (2021-2023, all hours).

NOTE: This dataset lacks a time-of-day field, so arrests cannot be
filtered to nighttime. All arrests are included. Column names reflect
this (e.g., arrest_count, not nighttime_arrest_count).

Dataset ID: 8h9b-rp9u
Has lat/lon but no community district — uses spatial join.
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

DATASET_ID = "8h9b-rp9u"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "nypd_arrests"
OUTPUT_PATH = REPORTS_DIR / "nighttime_arrests_cd.csv"
PAGE_SIZE = 50000


def fetch_arrests(params: dict, logger) -> pd.DataFrame:
    """Fetch nighttime arrest records with pagination."""
    cache_path = CACHE_DIR / "nighttime_arrests_raw.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    year_start = params["time_windows"]["primary"]["year_start"]
    year_end = params["time_windows"]["primary"]["year_end"]

    # arrest_date is a date field, no time component available directly.
    # The dataset doesn't have arrest_time. We'll fetch all arrests and note
    # this limitation — arrests don't have time-of-day in this dataset.
    # HOWEVER, checking if there's a time field...
    # The dataset has no time field, only arrest_date.
    # We'll fetch all arrests and compute total rates (not strictly nighttime).
    # This is noted as a limitation.

    where_clause = (
        f"arrest_date >= '{year_start}-01-01T00:00:00' "
        f"AND arrest_date < '{year_end + 1}-01-01T00:00:00' "
        f"AND latitude IS NOT NULL"
    )

    select_fields = (
        "arrest_key, arrest_date, pd_desc, ofns_desc, law_cat_cd, "
        "arrest_boro, arrest_precinct, latitude, longitude"
    )

    all_rows = []
    offset = 0

    while True:
        api_params = {
            "$where": where_clause,
            "$select": select_fields,
            "$order": "arrest_key",
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }

        logger.info(f"Fetching arrests page at offset {offset}...")
        for attempt in range(3):
            try:
                resp = requests.get(BASE_URL, params=api_params, timeout=300)
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                if attempt < 2:
                    logger.warning(f"  Retry {attempt + 1}/3 after timeout")
                    import time; time.sleep(5)
                else:
                    raise

        if not data:
            break

        all_rows.extend(data)
        logger.info(f"  Got {len(data)} rows (total: {len(all_rows)})")

        if len(data) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    df = pd.DataFrame(all_rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached {len(df)} rows")

    return df


def process(df: pd.DataFrame, params: dict, logger) -> pd.DataFrame:
    """Spatial join and compute rates."""
    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    cd = gpd.read_file(GEO_DIR / "cd59.geojson")
    cd["boro_cd"] = cd["boro_cd"].astype(int)
    cd = cd[cd["boro_cd"].isin(VALID_CDS)][["boro_cd", "geometry"]]

    joined = gpd.sjoin(gdf, cd, how="inner", predicate="within")
    logger.info(f"Spatial join: {len(df)} arrests -> {len(joined)} matched")

    jdf = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))

    # Aggregate by CD
    total = jdf.groupby("boro_cd").size().reset_index(name="arrest_count")

    # Felony arrests
    felony = jdf[jdf["law_cat_cd"] == "F"].groupby("boro_cd").size().reset_index(name="felony_arrests")

    # Drug offenses
    drug_mask = jdf["ofns_desc"].str.contains("DRUG|CONTROLLED SUBSTANCE|MARIJUANA", case=False, na=False)
    drug = jdf[drug_mask].groupby("boro_cd").size().reset_index(name="drug_arrests")

    # DUI/DWI
    dui_mask = jdf["ofns_desc"].str.contains("INTOXICATED|DWI|DUI", case=False, na=False) | \
               jdf["pd_desc"].str.contains("INTOXICATED|DWI|DUI", case=False, na=False)
    dui = jdf[dui_mask].groupby("boro_cd").size().reset_index(name="dui_arrests")

    result = total
    for sub in [felony, drug, dui]:
        result = result.merge(sub, on="boro_cd", how="left")
    result = result.fillna(0)

    # Population rates
    master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    result = result.merge(pop, on="boro_cd", how="left")

    years = params["time_windows"]["primary"]["year_end"] - params["time_windows"]["primary"]["year_start"] + 1

    pop_per_1k = result["population"] / 1000
    result["arrest_rate_per_1k"] = (
        result["arrest_count"] / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)
    result["felony_arrest_rate_per_1k"] = (
        result["felony_arrests"] / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)
    result["drug_arrest_rate_per_1k"] = (
        result["drug_arrests"] / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)
    result["dui_arrest_rate_per_1k"] = (
        result["dui_arrests"] / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    # Arrest-to-crime ratio (NOTE: arrests are all-hours, crimes are nighttime-only;
    # this ratio is an approximation since arrest data lacks time-of-day)
    crime_path = REPORTS_DIR / "nighttime_crime_cd.csv"
    if crime_path.exists():
        crime = pd.read_csv(crime_path)
        if "nighttime_crimes_total" in crime.columns:
            crime_col = "nighttime_crimes_total"
        elif "total_crimes" in crime.columns:
            crime_col = "total_crimes"
        elif "crime_count" in crime.columns:
            crime_col = "crime_count"
        else:
            crime_col = None

        if crime_col:
            crime["boro_cd"] = crime["boro_cd"].astype(int)
            result = result.merge(crime[["boro_cd", crime_col]], on="boro_cd", how="left")
            result["all_hours_arrest_to_night_crime_ratio"] = (
                result["arrest_count"] / result[crime_col]
            ).fillna(0).replace([float("inf"), float("-inf")], 0)

    logger.log_metrics({
        "total_arrests": int(result["arrest_count"].sum()),
        "total_felony": int(result["felony_arrests"].sum()),
        "total_drug": int(result["drug_arrests"].sum()),
        "total_dui": int(result["dui_arrests"].sum()),
    })

    return result


def main():
    with get_logger("21_collect_arrests") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        logger.warning(
            "NOTE: NYPD arrests dataset does not include time-of-day. "
            "These are ALL arrests 2021-2023, not strictly nighttime."
        )

        raw_df = fetch_arrests(params, logger)
        result = process(raw_df, params, logger)

        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population")]
        result[fill_cols] = result[fill_cols].fillna(0)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)
        logger.info(f"Saved {len(result)} rows to {OUTPUT_PATH}")

        print(f"\nNYPD arrests (2021-2023, all hours — no time field available):")
        print(f"  Total arrests: {int(result['arrest_count'].sum()):,}")
        print(f"  Felony: {int(result['felony_arrests'].sum()):,}")
        print(f"  Drug: {int(result['drug_arrests'].sum()):,}")
        print(f"  DUI: {int(result['dui_arrests'].sum()):,}")
        print(f"  Mean rate/1K/yr: {result['arrest_rate_per_1k'].mean():.2f}")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
