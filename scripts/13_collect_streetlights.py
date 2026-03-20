"""
13_collect_streetlights.py — Streetlight proxy metrics from 311 complaints and VIIRS.

The NYC DOT streetlight inventory is not available via Socrata API (datasets exist
in catalog but return "Not found"). As a proxy we use:
1. 311 "Street Light Condition" complaints (broken streetlight rate per CD)
2. VIIRS satellite radiance (already in master_analysis_df) as a lighting proxy
3. Area-normalized complaint rates as an inverse density proxy

If the streetlight inventory becomes API-accessible, this script should be updated.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import atomic_write_df, read_yaml, read_df
from nightscape.logging_utils import get_logger
from nightscape.paths import REPORTS_DIR, CONFIG_DIR, GEO_DIR
from nightscape.qa import VALID_CDS, filter_standard_cds
from nightscape.schemas import ensure_boro_cd_dtype

OUTPUT_PATH = REPORTS_DIR / "streetlight_proxy_cd.csv"


def main():
    with get_logger("13_collect_streetlights") as logger:
        params = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(params)

        # Load 311 data — broken streetlight complaints already extracted
        path_311 = REPORTS_DIR / "nighttime_311_all_cd.csv"
        if not path_311.exists():
            logger.error("Run 11_collect_311_all.py first")
            return

        df_311 = pd.read_csv(path_311)

        # Load master analysis df for VIIRS light and area data
        master = read_df(REPORTS_DIR / "master_analysis_df.parquet")

        # Load CD areas
        import geopandas as gpd
        cd_geo = gpd.read_file(GEO_DIR / "cd59.geojson")
        cd_geo["boro_cd"] = cd_geo["boro_cd"].astype(int)
        # Compute area in km2 using projected CRS
        cd_proj = cd_geo.to_crs("EPSG:2263")
        # EPSG:2263 is in US survey feet; 1 sq foot = 9.2903e-8 km2
        cd_geo["area_km2"] = cd_proj.geometry.area * 9.2903e-8

        areas = cd_geo[["boro_cd", "area_km2"]].copy()

        # Build result
        result = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        result = result.merge(
            df_311[["boro_cd", "streetlight_count", "broken_streetlight_rate"]].rename(
                columns={"streetlight_count": "streetlight_complaints"}
            ).assign(boro_cd=lambda x: x["boro_cd"].astype(int)),
            on="boro_cd", how="left"
        )
        result = result.merge(areas, on="boro_cd", how="left")

        # Complaints per km2 (proxy for streetlight issue density)
        years = params["time_windows"]["primary"]["year_end"] - params["time_windows"]["primary"]["year_start"] + 1
        result["streetlight_complaints_per_km2"] = (
            result["streetlight_complaints"] / years / result["area_km2"]
        ).replace([float("inf"), float("-inf")], 0).fillna(0)

        # Merge VIIRS radiance from master (lighting proxy)
        viirs_cols = [c for c in master.columns if "light" in c.lower() or "viirs" in c.lower() or "raw_light" in c.lower()]
        if viirs_cols:
            viirs_data = master[["boro_cd"] + viirs_cols].copy()
            viirs_data["boro_cd"] = viirs_data["boro_cd"].astype(int)
            result = result.merge(viirs_data, on="boro_cd", how="left")
            logger.info(f"Merged VIIRS columns: {viirs_cols}")
        else:
            logger.warning("No VIIRS/light columns found in master_analysis_df")

        result = result.fillna(0)
        result = ensure_boro_cd_dtype(result)
        result = filter_standard_cds(result)

        atomic_write_df(result, OUTPUT_PATH, index=False)
        logger.info(f"Saved {len(result)} rows to {OUTPUT_PATH}")

        print(f"\nStreetlight proxy metrics:")
        print(f"  Total 311 streetlight complaints: {int(result['streetlight_complaints'].sum()):,}")
        print(f"  Mean complaints/km2/yr: {result['streetlight_complaints_per_km2'].mean():.1f}")
        print(f"  VIIRS columns merged: {viirs_cols}")
        print(f"  NOTE: Actual streetlight inventory not available via API.")
        print(f"        Using 311 complaints + VIIRS radiance as proxies.")
        print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
