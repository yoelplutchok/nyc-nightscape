"""
35_collect_ems_response_times.py — EMS response times night vs day by borough.

Dataset ID: 76xm-jjuj (EMS Incident Dispatch Data)
Has incident_datetime, incident_response_seconds_qy, borough.
No lat/lon or CD — aggregates by borough, distributes to CDs.
"""

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

DATASET_ID = "76xm-jjuj"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "ems_response"
OUTPUT_PATH = REPORTS_DIR / "ems_response_cd.csv"

BORO_MAP = {
    "MANHATTAN": 1, "BRONX": 2, "BROOKLYN": 3,
    "QUEENS": 4, "RICHMOND / STATEN ISLAND": 5, "STATEN ISLAND": 5,
}


def fetch_ems_stats(params: dict, logger) -> pd.DataFrame:
    """Fetch night and day EMS response time stats by borough via server-side agg."""
    cache_path = CACHE_DIR / "ems_response_agg.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    year_start = params["time_windows"]["primary"]["year_start"]
    year_end = params["time_windows"]["primary"]["year_end"]

    all_results = []

    for period, hour_clause in [
        ("night", "(date_extract_hh(incident_datetime) >= 22 OR date_extract_hh(incident_datetime) < 7)"),
        ("day", "(date_extract_hh(incident_datetime) >= 7 AND date_extract_hh(incident_datetime) < 22)"),
    ]:
        where = (
            f"incident_datetime >= '{year_start}-01-01' "
            f"AND incident_datetime < '{year_end + 1}-01-01' "
            f"AND valid_incident_rspns_time_indc = 'Y' "
            f"AND {hour_clause}"
        )

        select = (
            "borough, "
            "count(*) as incident_count, "
            "avg(incident_response_seconds_qy) as mean_response_sec, "
            "avg(dispatch_response_seconds_qy) as mean_dispatch_sec"
        )

        api_params = {
            "$select": select,
            "$where": where,
            "$group": "borough",
            "$order": "borough",
            "$limit": 50000,
        }

        logger.info(f"Fetching EMS {period} response times...")
        resp = requests.get(BASE_URL, params=api_params, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        for row in data:
            row["period"] = period

        all_results.extend(data)
        logger.info(f"  {period}: {len(data)} borough rows")

    df = pd.DataFrame(all_results)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def process(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Pivot and distribute borough-level stats to CDs."""
    df = df.copy()
    for col in ["incident_count", "mean_response_sec", "mean_dispatch_sec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Map borough names to codes
    df["boro"] = df["borough"].map(BORO_MAP)
    df = df.dropna(subset=["boro"])
    df["boro"] = df["boro"].astype(int)

    # Pivot night/day
    night = df[df["period"] == "night"].copy()
    day = df[df["period"] == "day"].copy()

    night = night.rename(columns={
        "incident_count": "ems_incidents_night",
        "mean_response_sec": "ems_response_sec_night",
        "mean_dispatch_sec": "ems_dispatch_sec_night",
    })[["boro", "ems_incidents_night", "ems_response_sec_night", "ems_dispatch_sec_night"]]

    day = day.rename(columns={
        "incident_count": "ems_incidents_day",
        "mean_response_sec": "ems_response_sec_day",
        "mean_dispatch_sec": "ems_dispatch_sec_day",
    })[["boro", "ems_incidents_day", "ems_response_sec_day", "ems_dispatch_sec_day"]]

    boro_stats = night.merge(day, on="boro", how="outer")

    # Night/day response ratio
    boro_stats["ems_night_day_response_ratio"] = (
        boro_stats["ems_response_sec_night"] / boro_stats["ems_response_sec_day"]
    ).replace([float("inf"), float("-inf")], 1.0).fillna(1.0)

    # Convert response times to minutes
    boro_stats["ems_response_min_night"] = boro_stats["ems_response_sec_night"] / 60
    boro_stats["ems_response_min_day"] = boro_stats["ems_response_sec_day"] / 60

    # Distribute to CDs: each CD inherits its borough's stats.
    # NOTE: Response times are borough-level (not CD-level), so ems_response_min_night,
    # ems_response_min_day, and ems_night_day_response_ratio have only 5 unique values
    # across 59 CDs. Incident counts are estimated per CD using population share.
    all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
    all_cds["boro"] = all_cds["boro_cd"] // 100

    result = all_cds.merge(boro_stats, on="boro", how="left")

    # Per-capita incident rate using population
    master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    result = result.merge(pop, on="boro_cd", how="left")

    # Borough-level population for distributing incident counts
    boro_pop = result.groupby("boro")["population"].sum().reset_index()
    boro_pop.columns = ["boro", "boro_population"]
    result = result.merge(boro_pop, on="boro", how="left")

    result["pop_share"] = (
        result["population"] / result["boro_population"]
    ).replace([float("inf"), float("-inf")], 0).fillna(0)
    result["ems_incidents_night_est"] = (result["ems_incidents_night"] * result["pop_share"]).round(0)

    keep_cols = [
        "boro_cd", "ems_response_min_night", "ems_response_min_day",
        "ems_night_day_response_ratio", "ems_dispatch_sec_night",
        "ems_dispatch_sec_day", "ems_incidents_night_est",
    ]
    result = result[keep_cols].copy()
    fill_cols = [c for c in result.columns if c != "boro_cd"]
    result[fill_cols] = result[fill_cols].fillna(0)

    logger.log_metrics({
        "boroughs_covered": len(boro_stats),
        "mean_night_response_min": boro_stats["ems_response_min_night"].mean(),
        "mean_day_response_min": boro_stats["ems_response_min_day"].mean(),
        "mean_night_day_ratio": boro_stats["ems_night_day_response_ratio"].mean(),
    })

    return result


def main():
    with get_logger("35_collect_ems_response_times") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        raw_df = fetch_ems_stats(params, logger)
        result = process(raw_df, logger)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)

        print(f"\nEMS Response Times (2021-2023):")
        print(f"  Mean night response: {result['ems_response_min_night'].mean():.1f} min")
        print(f"  Mean day response: {result['ems_response_min_day'].mean():.1f} min")
        print(f"  Night/day ratio: {result['ems_night_day_response_ratio'].mean():.2f}")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
