"""
12_collect_fdny.py — Fetch nighttime FDNY fire incidents (2021-2023).

Dataset ID: 8m42-w767
Has communitydistrict field (e.g., "502") and incident_datetime.
Uses server-side aggregation.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import atomic_write_df, read_yaml, read_df
from nightscape.logging_utils import get_logger
from nightscape.paths import RAW_DIR, REPORTS_DIR, CONFIG_DIR
from nightscape.qa import VALID_CDS, filter_standard_cds
from nightscape.schemas import ensure_boro_cd_dtype

DATASET_ID = "8m42-w767"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "fdny_incidents"
OUTPUT_PATH = REPORTS_DIR / "nighttime_fdny_cd.csv"


def fetch_fdny_nighttime(params: dict, logger) -> pd.DataFrame:
    """Fetch nighttime FDNY incidents aggregated by CD and classification group."""
    cache_path = CACHE_DIR / "fdny_nighttime_raw.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    year_start = params["time_windows"]["primary"]["year_start"]
    year_end = params["time_windows"]["primary"]["year_end"]

    where_clause = (
        f"incident_datetime >= '{year_start}-01-01T00:00:00' "
        f"AND incident_datetime < '{year_end + 1}-01-01T00:00:00' "
        f"AND (date_extract_hh(incident_datetime) >= 22 "
        f"OR date_extract_hh(incident_datetime) < 7) "
        f"AND communitydistrict IS NOT NULL"
    )

    select_clause = (
        "communitydistrict, incident_classification_group, "
        "count(*) as incident_count"
    )

    api_params = {
        "$where": where_clause,
        "$select": select_clause,
        "$group": "communitydistrict, incident_classification_group",
        "$limit": 50000,
    }

    logger.info("Fetching nighttime FDNY incidents...")
    resp = requests.get(BASE_URL, params=api_params, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    logger.info(f"Received {len(data)} aggregated rows")
    df = pd.DataFrame(data)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def fetch_fdny_daytime(params: dict, logger) -> pd.DataFrame:
    """Fetch daytime FDNY totals for night/day ratio."""
    cache_path = CACHE_DIR / "fdny_daytime_raw.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached daytime data from {cache_path}")
        return pd.read_parquet(cache_path)

    year_start = params["time_windows"]["primary"]["year_start"]
    year_end = params["time_windows"]["primary"]["year_end"]

    where_clause = (
        f"incident_datetime >= '{year_start}-01-01T00:00:00' "
        f"AND incident_datetime < '{year_end + 1}-01-01T00:00:00' "
        f"AND date_extract_hh(incident_datetime) >= 7 "
        f"AND date_extract_hh(incident_datetime) < 22 "
        f"AND communitydistrict IS NOT NULL"
    )

    api_params = {
        "$where": where_clause,
        "$select": "communitydistrict, count(*) as day_count",
        "$group": "communitydistrict",
        "$limit": 50000,
    }

    logger.info("Fetching daytime FDNY incidents for night/day ratio...")
    resp = requests.get(BASE_URL, params=api_params, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def process(night_df: pd.DataFrame, day_df: pd.DataFrame, params: dict, logger) -> pd.DataFrame:
    """Aggregate and compute rates."""
    night_df = night_df.copy()
    night_df["incident_count"] = pd.to_numeric(night_df["incident_count"], errors="coerce").fillna(0).astype(int)
    night_df["boro_cd"] = pd.to_numeric(night_df["communitydistrict"], errors="coerce")
    night_df = night_df.dropna(subset=["boro_cd"])
    night_df["boro_cd"] = night_df["boro_cd"].astype(int)
    night_df = night_df[night_df["boro_cd"].isin(VALID_CDS)]

    # Total nighttime incidents per CD
    total_night = night_df.groupby("boro_cd")["incident_count"].sum().reset_index()
    total_night.columns = ["boro_cd", "fire_incidents_night"]

    # Structural fires (only actual structural fires, excluding non-structural)
    structural = night_df[night_df["incident_classification_group"] == "Structural Fires"]
    struct_agg = structural.groupby("boro_cd")["incident_count"].sum().reset_index()
    struct_agg.columns = ["boro_cd", "structural_fires_night"]

    # Gas leaks
    gas = night_df[night_df["incident_classification_group"].str.contains("Gas", case=False, na=False)]
    gas_agg = gas.groupby("boro_cd")["incident_count"].sum().reset_index()
    gas_agg.columns = ["boro_cd", "gas_leak_incidents_night"]

    # CO incidents
    co = night_df[night_df["incident_classification_group"].str.contains(r"Carbon Monoxide|\bCO\b", case=False, na=False)]
    co_agg = co.groupby("boro_cd")["incident_count"].sum().reset_index()
    co_agg.columns = ["boro_cd", "co_incidents_night"]

    # Merge
    result = total_night
    for sub in [struct_agg, gas_agg, co_agg]:
        result = result.merge(sub, on="boro_cd", how="left")
    result = result.fillna(0)

    # Daytime for ratio
    day_df = day_df.copy()
    day_df["day_count"] = pd.to_numeric(day_df["day_count"], errors="coerce").fillna(0).astype(int)
    day_df["boro_cd"] = pd.to_numeric(day_df["communitydistrict"], errors="coerce")
    day_df = day_df.dropna(subset=["boro_cd"])
    day_df["boro_cd"] = day_df["boro_cd"].astype(int)
    day_agg = day_df.groupby("boro_cd")["day_count"].sum().reset_index()
    result = result.merge(day_agg, on="boro_cd", how="left")
    result["day_count"] = result["day_count"].fillna(0)

    # Night/day ratio (adjusted for hours: 9 night hours vs 15 day hours)
    result["fire_night_day_ratio"] = (
        (result["fire_incidents_night"] / 9) / (result["day_count"] / 15)
    ).fillna(0).replace([float("inf"), float("-inf")], 0)

    # Population rates
    master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    result = result.merge(pop, on="boro_cd", how="left")

    years = params["time_windows"]["primary"]["year_end"] - params["time_windows"]["primary"]["year_start"] + 1
    pop_per_1k = result["population"] / 1000

    result["fire_incidents_night_per_1k"] = (
        result["fire_incidents_night"] / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    result["structural_fires_night_per_1k"] = (
        result["structural_fires_night"] / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    logger.log_metrics({
        "total_nighttime_incidents": int(result["fire_incidents_night"].sum()),
        "total_structural_fires": int(result["structural_fires_night"].sum()),
        "cds_with_data": len(result),
    })

    return result


def main():
    with get_logger("12_collect_fdny") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        night_df = fetch_fdny_nighttime(params, logger)
        day_df = fetch_fdny_daytime(params, logger)
        result = process(night_df, day_df, params, logger)

        # Ensure all 59 CDs
        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population")]
        result[fill_cols] = result[fill_cols].fillna(0)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)
        logger.info(f"Saved {len(result)} rows to {OUTPUT_PATH}")

        print(f"\nNighttime FDNY incidents (2021-2023):")
        print(f"  Total incidents: {int(result['fire_incidents_night'].sum()):,}")
        print(f"  Structural fires: {int(result['structural_fires_night'].sum()):,}")
        print(f"  CDs with data: {(result['fire_incidents_night'] > 0).sum()}/59")
        print(f"  Mean incidents/1K/yr: {result['fire_incidents_night_per_1k'].mean():.2f}")
        print(f"  Mean night/day ratio: {result['fire_night_day_ratio'].mean():.2f}")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
