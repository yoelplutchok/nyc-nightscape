"""
11_collect_311_all.py — Fetch nighttime non-noise 311 complaints (2021-2023).

Dataset ID: erm2-nwe9 (same as noise, different filters)
Uses server-side aggregation by complaint_type and community_board.
Excludes noise types (already have those from Night Signals).
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

DATASET_ID = "erm2-nwe9"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "311_all"
OUTPUT_PATH = REPORTS_DIR / "nighttime_311_all_cd.csv"

# Noise-related types to EXCLUDE (already captured in Night Signals data)
NOISE_TYPES = [
    "Noise - Residential",
    "Noise - Commercial",
    "Noise - Street/Sidewalk",
    "Noise - Vehicle",
    "Noise - Park",
    "Noise - Helicopter",
    "Noise",
    "Noise - House of Worship",
    "Collection Truck Noise",
]

# Key categories to track individually
KEY_CATEGORIES = {
    "homeless": ["Homeless Person Assistance", "Homeless Encampment"],
    "streetlight": ["Street Light Condition"],
    "drug_alcohol": ["Drug Activity", "Drinking"],
    "quality_of_life": [
        "Illegal Parking", "Blocked Driveway", "Rodent",
        "Unsanitary Condition", "Graffiti", "Urinating in Public",
        "Disorderly Youth", "Smoking",
    ],
}


def fetch_311_nighttime(params: dict, logger) -> pd.DataFrame:
    """Fetch non-noise 311 complaints at night, aggregated by complaint_type + community_board."""
    cache_path = CACHE_DIR / "311_all_nighttime_raw.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    year_start = params["time_windows"]["primary"]["year_start"]
    year_end = params["time_windows"]["primary"]["year_end"]

    # Build exclusion list for noise types
    noise_exclude = " AND ".join(
        [f"complaint_type != '{t}'" for t in NOISE_TYPES]
    )

    where_clause = (
        f"created_date >= '{year_start}-01-01T00:00:00' "
        f"AND created_date < '{year_end + 1}-01-01T00:00:00' "
        f"AND (date_extract_hh(created_date) >= 22 OR date_extract_hh(created_date) < 7) "
        f"AND community_board != 'Unspecified' "
        f"AND {noise_exclude}"
    )

    select_clause = (
        "complaint_type, community_board, count(*) as complaint_count"
    )

    api_params = {
        "$where": where_clause,
        "$select": select_clause,
        "$group": "complaint_type, community_board",
        "$limit": 50000,
    }

    logger.info("Fetching non-noise 311 nighttime complaints...")
    resp = requests.get(BASE_URL, params=api_params, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    logger.info(f"Received {len(data)} aggregated rows")
    df = pd.DataFrame(data)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached to {cache_path}")

    return df


def parse_community_board(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Parse community_board field (e.g., '01 MANHATTAN') into boro_cd."""
    boro_map = {
        "MANHATTAN": 1, "BRONX": 2, "BROOKLYN": 3, "QUEENS": 4, "STATEN ISLAND": 5,
    }

    df = df.copy()
    df["complaint_count"] = pd.to_numeric(df["complaint_count"], errors="coerce").fillna(0).astype(int)

    # Parse "01 MANHATTAN" -> boro_cd 101
    parsed = []
    for _, row in df.iterrows():
        cb = str(row.get("community_board", ""))
        parts = cb.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        try:
            cd_num = int(parts[0])
        except ValueError:
            continue
        boro_name = parts[1].strip().upper()
        if boro_name not in boro_map:
            continue
        boro_cd = boro_map[boro_name] * 100 + cd_num
        if boro_cd in VALID_CDS:
            parsed.append({
                "boro_cd": boro_cd,
                "complaint_type": row["complaint_type"],
                "complaint_count": row["complaint_count"],
            })

    result = pd.DataFrame(parsed)
    logger.info(f"Parsed {len(result)} valid rows from {len(df)} raw rows")
    return result


def aggregate_to_cd(df: pd.DataFrame, params: dict, logger) -> pd.DataFrame:
    """Aggregate complaints by CD and compute rates."""
    # Total non-noise complaints per CD
    total = df.groupby("boro_cd")["complaint_count"].sum().reset_index()
    total.columns = ["boro_cd", "non_noise_311_count"]

    # Category-specific counts
    for cat_name, cat_types in KEY_CATEGORIES.items():
        cat_df = df[df["complaint_type"].isin(cat_types)]
        cat_agg = cat_df.groupby("boro_cd")["complaint_count"].sum().reset_index()
        cat_agg.columns = ["boro_cd", f"{cat_name}_count"]
        total = total.merge(cat_agg, on="boro_cd", how="left")

    total = total.fillna(0)

    # Load population
    master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    total = total.merge(pop, on="boro_cd", how="left")

    years = params["time_windows"]["primary"]["year_end"] - params["time_windows"]["primary"]["year_start"] + 1

    # Rates per 1K pop per year
    pop_per_1k = total["population"] / 1000
    total["non_noise_311_per_1k"] = (
        total["non_noise_311_count"] / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    total["homeless_per_1k"] = (
        total["homeless_count"] / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    total["broken_streetlight_rate"] = (
        total["streetlight_count"] / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    total["drug_alcohol_complaint_rate"] = (
        total["drug_alcohol_count"] / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    total["quality_of_life_rate"] = (
        total["quality_of_life_count"] / years / pop_per_1k
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    logger.log_metrics({
        "total_non_noise_complaints": int(total["non_noise_311_count"].sum()),
        "unique_complaint_types": df["complaint_type"].nunique(),
        "cds_with_data": len(total),
    })

    return total


def main():
    with get_logger("11_collect_311_all") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        raw_df = fetch_311_nighttime(params, logger)
        df = parse_community_board(raw_df, logger)
        result = aggregate_to_cd(df, params, logger)

        # Ensure all 59 CDs
        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population")]
        result[fill_cols] = result[fill_cols].fillna(0)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)
        logger.info(f"Saved {len(result)} rows to {OUTPUT_PATH}")

        print(f"\nNighttime non-noise 311 complaints (2021-2023):")
        print(f"  Total complaints: {int(result['non_noise_311_count'].sum()):,}")
        print(f"  CDs with data: {(result['non_noise_311_count'] > 0).sum()}/59")
        print(f"  Mean rate/1K/yr: {result['non_noise_311_per_1k'].mean():.2f}")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
