"""
32_extract_helicopter_noise.py — Extract helicopter noise complaints from 311 data.

Dataset ID: erm2-nwe9 (NYC 311 Service Requests)
Queries for complaint_type = 'Noise - Helicopter' with nighttime filter (22:00-06:59).
Also fetches total nighttime noise complaints to compute helicopter share.
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

DATASET_ID = "erm2-nwe9"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "helicopter_noise"
OUTPUT_PATH = REPORTS_DIR / "helicopter_noise_cd.csv"

BORO_MAP = {
    "MANHATTAN": 1,
    "BRONX": 2,
    "BROOKLYN": 3,
    "QUEENS": 4,
    "STATEN ISLAND": 5,
}


def build_nighttime_where(year_start: int, year_end: int) -> str:
    """Build the common nighttime + date-range WHERE clause."""
    return (
        f"created_date >= '{year_start}-01-01T00:00:00' "
        f"AND created_date < '{year_end + 1}-01-01T00:00:00' "
        f"AND (date_extract_hh(created_date) >= 22 OR date_extract_hh(created_date) < 7) "
        f"AND community_board != 'Unspecified'"
    )


def fetch_helicopter_counts(params: dict, logger) -> pd.DataFrame:
    """Fetch nighttime helicopter noise complaints aggregated by community_board."""
    cache_path = CACHE_DIR / "helicopter_noise_agg.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached helicopter data from {cache_path}")
        return pd.read_parquet(cache_path)

    year_start = params["time_windows"]["primary"]["year_start"]
    year_end = params["time_windows"]["primary"]["year_end"]

    where_clause = (
        build_nighttime_where(year_start, year_end)
        + " AND complaint_type = 'Noise - Helicopter'"
    )

    api_params = {
        "$select": "community_board, count(*) as cnt",
        "$where": where_clause,
        "$group": "community_board",
        "$limit": 50000,
    }

    logger.info("Fetching nighttime helicopter noise complaints...")
    resp = requests.get(BASE_URL, params=api_params, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    logger.info(f"Received {len(data)} aggregated rows for helicopter noise")
    df = pd.DataFrame(data)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached to {cache_path}")

    return df


def fetch_total_noise_counts(params: dict, logger) -> pd.DataFrame:
    """Fetch total nighttime noise complaints (all noise types) aggregated by community_board."""
    cache_path = CACHE_DIR / "total_noise_agg.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached total noise data from {cache_path}")
        return pd.read_parquet(cache_path)

    year_start = params["time_windows"]["primary"]["year_start"]
    year_end = params["time_windows"]["primary"]["year_end"]

    where_clause = (
        build_nighttime_where(year_start, year_end)
        + " AND complaint_type LIKE 'Noise%'"
    )

    api_params = {
        "$select": "community_board, count(*) as cnt",
        "$where": where_clause,
        "$group": "community_board",
        "$limit": 50000,
    }

    logger.info("Fetching total nighttime noise complaints...")
    resp = requests.get(BASE_URL, params=api_params, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    logger.info(f"Received {len(data)} aggregated rows for total noise")
    df = pd.DataFrame(data)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached to {cache_path}")

    return df


def parse_community_board(df: pd.DataFrame, count_col: str, logger) -> pd.DataFrame:
    """Parse community_board field (e.g., '01 MANHATTAN') into boro_cd and sum counts."""
    df = df.copy()
    df[count_col] = pd.to_numeric(df["cnt"], errors="coerce").fillna(0).astype(int)

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
        if boro_name not in BORO_MAP:
            continue
        boro_cd = BORO_MAP[boro_name] * 100 + cd_num
        if boro_cd in VALID_CDS:
            parsed.append({
                "boro_cd": boro_cd,
                count_col: row[count_col],
            })

    result = pd.DataFrame(parsed)
    if result.empty:
        result = pd.DataFrame(columns=["boro_cd", count_col])
    else:
        # Sum in case of duplicate boro_cd rows
        result = result.groupby("boro_cd", as_index=False)[count_col].sum()

    logger.info(f"Parsed {len(result)} valid CD rows for {count_col}")
    return result


def compute_rates(heli_df: pd.DataFrame, total_df: pd.DataFrame,
                  params: dict, logger) -> pd.DataFrame:
    """Merge helicopter + total noise, compute per-capita rates and shares."""
    result = heli_df.merge(total_df, on="boro_cd", how="outer").fillna(0)

    # Load population from master analysis df
    master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    result = result.merge(pop, on="boro_cd", how="left")

    years = (
        params["time_windows"]["primary"]["year_end"]
        - params["time_windows"]["primary"]["year_start"]
        + 1
    )

    # Per-1K-population annual rate
    result["helicopter_complaints_night_per_1k"] = (
        (result["helicopter_complaints_night"] / years)
        / (result["population"] / 1000)
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    # Helicopter as percentage of all nighttime noise
    result["helicopter_pct_of_noise"] = (
        result["helicopter_complaints_night"]
        / result["total_noise_complaints_night"].replace(0, float("nan"))
        * 100
    ).replace([float("inf"), float("-inf")], 0).fillna(0).round(2)

    logger.log_metrics({
        "total_helicopter_complaints": int(result["helicopter_complaints_night"].sum()),
        "total_noise_complaints": int(result["total_noise_complaints_night"].sum()),
        "citywide_helicopter_pct": round(
            result["helicopter_complaints_night"].sum()
            / max(result["total_noise_complaints_night"].sum(), 1) * 100, 2
        ),
        "cds_with_helicopter_complaints": int(
            (result["helicopter_complaints_night"] > 0).sum()
        ),
    })

    return result


def main():
    with get_logger("32_extract_helicopter_noise") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        # Fetch helicopter noise complaints
        heli_raw = fetch_helicopter_counts(params, logger)
        heli_cd = parse_community_board(
            heli_raw, "helicopter_complaints_night", logger
        )

        # Fetch total noise complaints
        total_raw = fetch_total_noise_counts(params, logger)
        total_cd = parse_community_board(
            total_raw, "total_noise_complaints_night", logger
        )

        # Compute rates and shares
        result = compute_rates(heli_cd, total_cd, params, logger)

        # Ensure all 59 CDs present
        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population")]
        result[fill_cols] = result[fill_cols].fillna(0)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)
        logger.info(f"Saved {len(result)} rows to {OUTPUT_PATH}")

        heli_total = int(result["helicopter_complaints_night"].sum())
        noise_total = int(result["total_noise_complaints_night"].sum())
        citywide_pct = heli_total / max(noise_total, 1) * 100

        print(f"\nNighttime helicopter noise complaints (2021-2023):")
        print(f"  Helicopter complaints: {heli_total:,}")
        print(f"  Total noise complaints: {noise_total:,}")
        print(f"  Citywide helicopter share: {citywide_pct:.2f}%")
        print(f"  CDs with helicopter complaints: "
              f"{(result['helicopter_complaints_night'] > 0).sum()}/59")
        print(f"  Mean rate/1K/yr: "
              f"{result['helicopter_complaints_night_per_1k'].mean():.3f}")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
