"""
27_collect_commercial_waste.py — Map commercial waste zones to CDs.

Dataset ID: 8ev8-jjxq (DSNY Commercial Waste Zones)
Has multipolygon geometry and CD references in zone descriptions.
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

DATASET_ID = "8ev8-jjxq"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
CACHE_DIR = RAW_DIR / "commercial_waste"
OUTPUT_PATH = REPORTS_DIR / "commercial_waste_cd.csv"


def fetch_waste_zones(logger) -> gpd.GeoDataFrame:
    """Fetch commercial waste zone polygons."""
    cache_path = CACHE_DIR / "waste_zones_raw.geojson"
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return gpd.read_file(cache_path)

    # Fetch as GeoJSON
    url = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.geojson?$limit=50000"
    logger.info("Fetching commercial waste zones...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        f.write(resp.text)

    gdf = gpd.read_file(cache_path)
    logger.info(f"Got {len(gdf)} waste zones")
    return gdf


def process(waste_gdf: gpd.GeoDataFrame, params: dict, logger) -> pd.DataFrame:
    """Compute overlap between waste zones and CDs."""
    # Load CD boundaries
    cd = gpd.read_file(GEO_DIR / "cd59.geojson")
    cd["boro_cd"] = cd["boro_cd"].astype(int)
    cd = cd[cd["boro_cd"].isin(VALID_CDS)][["boro_cd", "geometry"]]

    # Ensure same CRS
    if waste_gdf.crs != cd.crs:
        waste_gdf = waste_gdf.to_crs(cd.crs)

    # Compute area overlap: what fraction of each CD is covered by waste zones
    cd_proj = cd.to_crs("EPSG:2263")
    cd_proj["cd_area"] = cd_proj.geometry.area

    waste_proj = waste_gdf.to_crs("EPSG:2263")

    results = []
    for _, cd_row in cd_proj.iterrows():
        boro_cd = cd_row["boro_cd"]
        cd_geom = cd_row["geometry"]
        cd_area = cd_row["cd_area"]

        # Find overlapping waste zones
        overlapping = waste_proj[waste_proj.intersects(cd_geom)]
        n_zones = len(overlapping)

        if n_zones > 0:
            # Compute total overlap area
            overlap_area = sum(
                cd_geom.intersection(wz_geom).area
                for wz_geom in overlapping.geometry
            )
            pct_covered = (overlap_area / cd_area * 100) if cd_area > 0 else 0
        else:
            overlap_area = 0
            pct_covered = 0

        # Count unique carters (from zone names)
        zone_names = overlapping["zone"].tolist() if "zone" in overlapping.columns else []

        results.append({
            "boro_cd": boro_cd,
            "n_waste_zones": n_zones,
            "commercial_waste_zone_pct": round(pct_covered, 1),
            "waste_zone_names": ", ".join(zone_names) if zone_names else "",
        })

    result = pd.DataFrame(results)

    # Population merge
    master = read_df(REPORTS_DIR / "master_analysis_df.parquet")
    pop = master[["boro_cd", "population"]].copy()
    pop["boro_cd"] = pop["boro_cd"].astype(int)
    result = result.merge(pop, on="boro_cd", how="left")

    logger.log_metrics({
        "total_waste_zones": len(waste_gdf),
        "cds_with_coverage": (result["commercial_waste_zone_pct"] > 0).sum(),
        "mean_coverage_pct": result["commercial_waste_zone_pct"].mean(),
    })

    return result


def main():
    with get_logger("27_collect_commercial_waste") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        waste_gdf = fetch_waste_zones(logger)
        result = process(waste_gdf, params, logger)

        # Ensure all 59 CDs present
        all_cds = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = all_cds.merge(result, on="boro_cd", how="left")
        fill_cols = [c for c in result.columns if c not in ("boro_cd", "population")]
        result[fill_cols] = result[fill_cols].fillna(0)

        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)

        print(f"\nCommercial waste zones:")
        print(f"  Total zones: {len(waste_gdf)}")
        print(f"  CDs with coverage: {(result['commercial_waste_zone_pct'] > 0).sum()}/59")
        print(f"  Mean coverage: {result['commercial_waste_zone_pct'].mean():.1f}%")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
