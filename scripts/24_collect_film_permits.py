"""
24_collect_film_permits.py — Fetch nighttime film/TV shoot permits (2021-2023).

Dataset ID: tg4x-b46p
Has communityboard_s field and start/end datetimes.
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

DATASET_ID = "tg4x-b46p"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "film_permits"
OUTPUT_PATH = REPORTS_DIR / "night_film_permits_cd.csv"
PAGE_SIZE = 50000

BORO_MAP = {
    "Manhattan": 1, "Bronx": 2, "Brooklyn": 3, "Queens": 4, "Staten Island": 5,
}


def fetch_film_permits(params: dict, logger) -> pd.DataFrame:
    """Fetch film permits that extend into nighttime hours."""
    cache_path = CACHE_DIR / "film_permits_night_raw.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    year_start = params["time_windows"]["primary"]["year_start"]
    year_end = params["time_windows"]["primary"]["year_end"]

    # Night shoots: end time after 10PM or start time before 7AM
    where_clause = (
        f"startdatetime >= '{year_start}-01-01T00:00:00' "
        f"AND startdatetime < '{year_end + 1}-01-01T00:00:00' "
        f"AND (date_extract_hh(enddatetime) >= 22 "
        f"OR date_extract_hh(startdatetime) >= 22 "
        f"OR date_extract_hh(startdatetime) < 7 "
        f"OR date_extract_hh(enddatetime) < 7)"
    )

    all_rows = []
    offset = 0

    while True:
        api_params = {
            "$where": where_clause,
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }

        logger.info(f"Fetching film permits page at offset {offset}...")
        resp = requests.get(BASE_URL, params=api_params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)
        if len(data) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    logger.info(f"Total night film permits fetched: {len(all_rows)}")
    df = pd.DataFrame(all_rows)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def process(df: pd.DataFrame, params: dict, logger) -> pd.DataFrame:
    """Parse community boards and aggregate."""
    records = []
    for _, row in df.iterrows():
        borough = row.get("borough", "")
        cb_str = str(row.get("communityboard_s", ""))
        if not borough or borough not in BORO_MAP:
            continue

        boro_num = BORO_MAP[borough]
        # communityboard_s can be "5," or "1, 2," — multiple CDs
        for part in cb_str.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                cd_num = int(part)
                boro_cd = boro_num * 100 + cd_num
                if boro_cd in VALID_CDS:
                    records.append({"boro_cd": boro_cd, "event_id": row.get("eventid", "")})
            except ValueError:
                continue

    parsed = pd.DataFrame(records)
    logger.info(f"Parsed {len(parsed)} permit-CD combinations from {len(df)} permits")

    agg = parsed.groupby("boro_cd")["event_id"].nunique().reset_index()
    agg.columns = ["boro_cd", "night_film_permits"]

    # Population and area
    master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    agg = agg.merge(pop, on="boro_cd", how="left")

    years = params["time_windows"]["primary"]["year_end"] - params["time_windows"]["primary"]["year_start"] + 1
    agg["night_film_permits_per_year"] = agg["night_film_permits"] / years
    agg["film_activity_per_1k"] = (
        agg["night_film_permits"] / years / (agg["population"] / 1000)
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    logger.log_metrics({
        "total_night_film_permits": int(agg["night_film_permits"].sum()),
        "cds_with_permits": len(agg),
    })

    return agg


def main():
    with get_logger("24_collect_film_permits") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        raw_df = fetch_film_permits(params, logger)
        result = process(raw_df, params, logger)

        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population")]
        result[fill_cols] = result[fill_cols].fillna(0)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)

        print(f"\nNight film/TV shoot permits (2021-2023):")
        print(f"  Total permits: {int(result['night_film_permits'].sum()):,}")
        print(f"  CDs with permits: {(result['night_film_permits'] > 0).sum()}/59")
        print(f"  Mean permits/yr: {result['night_film_permits_per_year'].mean():.1f}")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
