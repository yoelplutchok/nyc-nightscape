"""
34_collect_linknyc.py — Fetch WiFi/LinkNYC hotspot locations and compute density per CD.

Dataset ID: yjub-udmw (NYC WiFi Hotspot Locations)
Has borocd field directly — no spatial join needed.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import atomic_write_df, read_yaml, read_df
from nightscape.logging_utils import get_logger
from nightscape.paths import RAW_DIR, REPORTS_DIR, CONFIG_DIR, GEO_DIR
from nightscape.qa import VALID_CDS, filter_standard_cds
from nightscape.schemas import ensure_boro_cd_dtype

DATASET_ID = "yjub-udmw"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "linknyc"
OUTPUT_PATH = REPORTS_DIR / "linknyc_wifi_cd.csv"
PAGE_SIZE = 50000


def fetch_hotspots(logger) -> pd.DataFrame:
    """Fetch all WiFi hotspot locations via Socrata JSON API."""
    cache_path = CACHE_DIR / "wifi_hotspots_raw.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    all_rows = []
    offset = 0

    while True:
        api_params = {
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }

        logger.info(f"Fetching WiFi hotspots page at offset {offset}...")
        resp = requests.get(BASE_URL, params=api_params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)
        logger.info(f"  Got {len(data)} rows (total: {len(all_rows)})")

        if len(data) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    logger.info(f"Total WiFi hotspots fetched: {len(all_rows)}")
    df = pd.DataFrame(all_rows)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def process(df: pd.DataFrame, params: dict, logger) -> pd.DataFrame:
    """Aggregate by CD using the borocd field."""
    df = df.copy()

    # Parse borocd (format: "412.00000000000" → 412)
    df["boro_cd"] = pd.to_numeric(df["borocd"], errors="coerce")
    df = df.dropna(subset=["boro_cd"])
    df["boro_cd"] = df["boro_cd"].astype(int)
    df = df[df["boro_cd"].isin(VALID_CDS)]

    logger.info(f"Hotspots with valid CD: {len(df)}")

    # Total WiFi hotspots per CD
    total = df.groupby("boro_cd").size().reset_index(name="wifi_hotspots")

    # LinkNYC kiosks — filter by provider or type containing "Link"
    linknyc_mask = pd.Series(False, index=df.index)
    for col in ["provider", "type", "ssid", "name"]:
        if col in df.columns:
            linknyc_mask = linknyc_mask | df[col].astype(str).str.contains(
                r"(?i)link", na=False
            )
    linknyc = df[linknyc_mask]
    ln_agg = linknyc.groupby("boro_cd").size().reset_index(name="linknyc_kiosks")

    logger.info(f"LinkNYC kiosks: {ln_agg['linknyc_kiosks'].sum() if len(ln_agg) else 0}")

    result = total.merge(ln_agg, on="boro_cd", how="left")
    result = result.fillna(0)

    # CD areas for density
    cd_geo = gpd.read_file(GEO_DIR / "cd59.geojson")
    cd_geo["boro_cd"] = cd_geo["boro_cd"].astype(int)
    cd_proj = cd_geo.to_crs("EPSG:2263")
    cd_geo["area_km2"] = cd_proj.geometry.area * 9.2903e-8
    result = result.merge(cd_geo[["boro_cd", "area_km2"]], on="boro_cd", how="left")

    result["wifi_hotspots_per_km2"] = (
        result["wifi_hotspots"] / result["area_km2"]
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    # Population for per-1k rates
    master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    result = result.merge(pop, on="boro_cd", how="left")

    result["linknyc_kiosks_per_1k"] = (
        result["linknyc_kiosks"] / (result["population"] / 1000)
    ).replace([float("inf"), float("-inf")], 0).fillna(0)

    logger.log_metrics({
        "total_wifi_hotspots": int(result["wifi_hotspots"].sum()),
        "total_linknyc_kiosks": int(result["linknyc_kiosks"].sum()),
        "mean_hotspots_per_km2": result["wifi_hotspots_per_km2"].mean(),
    })

    return result


def main():
    with get_logger("34_collect_linknyc") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        raw_df = fetch_hotspots(logger)
        result = process(raw_df, params, logger)

        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population", "area_km2")]
        result[fill_cols] = result[fill_cols].fillna(0)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)

        print(f"\nWiFi / LinkNYC hotspot density:")
        print(f"  Total WiFi hotspots: {int(result['wifi_hotspots'].sum()):,}")
        print(f"  Total LinkNYC kiosks: {int(result['linknyc_kiosks'].sum()):,}")
        print(f"  Mean hotspots/km2: {result['wifi_hotspots_per_km2'].mean():.1f}")
        print(f"  CDs with hotspots: {(result['wifi_hotspots'] > 0).sum()}/59")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
