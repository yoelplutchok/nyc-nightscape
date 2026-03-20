"""
50_build_master_nightscape.py — Build the master nightscape dataframe.

Merges all CD-level reports (legacy + new) into a single 59-row dataframe
with 60-80+ variables across 8 dimensions.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nightscape.io_utils import atomic_write_df, read_df
from nightscape.logging_utils import get_logger
from nightscape.paths import REPORTS_DIR, GEO_DIR
from nightscape.qa import VALID_CDS, filter_standard_cds
from nightscape.schemas import ensure_boro_cd_dtype

OUTPUT_PATH = REPORTS_DIR / "master_nightscape_df.parquet"
OUTPUT_CSV = REPORTS_DIR / "master_nightscape_df.csv"


# ── Source definitions ──────────────────────────────────────────────
# Each entry: (filename, list of columns to keep)
# "boro_cd" is always the join key and added automatically.

SOURCES = [
    # ── Legacy master (demographics, environmental, noise baselines) ──
    ("master_analysis_df.parquet", [
        "borough", "population", "area_km2", "pop_density",
        "poverty_rate", "pct_nonhisp_black", "pct_hispanic", "rent_burden_rate",
        "radiance_raw", "z_light", "z_noise_obj", "z_air", "z_heat",
        "noise_obj_db_mean", "dot_exposure_index", "dot_pct_above_65dB",
        "rate_per_1k_pop", "rate_per_km2",  # 311 noise
        "late_night_share", "weekend_uplift", "warm_season_ratio",
        "no2_mean_primary", "pm25_mean_primary", "tmin_mean_primary",
    ]),

    # ── Legacy reports ──
    ("nighttime_crime_cd.csv", [
        "nighttime_crimes_total", "crimes_per_year",
        "felonies", "misdemeanors", "felony_share",
        "crime_rate_per_1k", "violent_crimes", "violent_rate_per_1k",
    ]),
    ("shots_fired_cd.csv", [
        "shots_total", "shots_per_year", "shots_nighttime",
        "shots_nighttime_share", "shots_rate_per_1k",
    ]),
    ("liquor_license_cd.csv", [
        "total_licenses", "on_premises_count",
        "license_rate_per_1k_pop", "on_premises_rate_per_1k_pop",
    ]),
    ("siren_density_cd.csv", [
        "ems_total", "ems_nighttime", "fire_total", "fire_nighttime",
        "siren_density_per_km2", "siren_rate_per_1k",
    ]),
    ("ecb_noise_violations_cd.csv", [
        "total_violations", "dep_violations", "nypd_violations",
        "violation_rate_per_1k",
    ]),
    ("ahv_permits_cd.csv", [
        "ahv_total_permits", "ahv_nighttime_permits", "ahv_nighttime_share",
    ]),
    ("dob_construction_density_cd.csv", [
        "total_permits", "permits_per_year",
        "construction_density_per_km2",
    ]),
    ("dot_traffic_cd.csv", [
        "mean_volume", "median_volume",
    ]),
    ("pluto_cd_metrics.csv", [
        "mean_year_built", "mean_floors", "total_res_units",
        "pct_residential", "pct_commercial", "pct_mixed_use",
        "mean_lot_area_sqft",
    ]),
    ("chs_cd_estimates.csv", [
        "depression", "heavy_drinking", "obesity",
        "poor_health", "sleep_trouble",
    ]),
    ("nycha_noise_exposure.csv", [
        "nycha_pop_share", "nycha_buildings",
    ]),
    ("descriptor_rates_cd.csv", [
        "Loud Music/Party", "Loud Talking", "Banging/Pounding",
        "Car/Truck Music", "Car/Truck Horn", "Engine Idling",
        "Construction", "Barking Dog",
    ]),
    ("resolution_equity_cd.csv", [
        "res_action_taken_share", "res_summons_issued_share",
    ]),
    ("response_time_cd.csv", [
        "median_response_hours",
    ]),

    # ── Tier 1 new data ──
    ("nighttime_crashes_cd.csv", [
        "crash_count", "persons_injured", "persons_killed",
        "ped_injured", "cyclist_injured",
        "crash_rate_per_1k", "injury_rate_per_1k",
        "night_crash_severity",
    ]),
    ("nighttime_311_all_cd.csv", [
        "non_noise_311_count", "homeless_count", "streetlight_count",
        "drug_alcohol_count", "quality_of_life_count",
        "non_noise_311_per_1k", "homeless_per_1k",
    ]),
    ("nighttime_fdny_cd.csv", [
        "fire_incidents_night", "structural_fires_night",
        "fire_night_day_ratio", "fire_incidents_night_per_1k",
    ]),
    ("streetlight_proxy_cd.csv", [
        "streetlight_complaints", "streetlight_complaints_per_km2",
    ]),
    ("late_night_subway_cd.csv", [
        "late_night_ridership", "subway_stations",
        "pct_ridership_late_night",
        "late_night_entries_per_1k", "subway_stations_per_km2",
        "transit_desert_flag",
    ]),

    # ── Tier 2 new data ──
    ("nighttime_tlc_trips_cd.csv", [
        "taxi_trips_night", "rideshare_trips_night", "total_ride_trips_night",
        "taxi_pickups_night_per_1k", "rideshare_pickups_night_per_1k",
        "total_ride_pickups_night_per_1k", "rideshare_to_taxi_ratio",
    ]),
    ("nighttime_arrests_cd.csv", [
        "arrest_count", "felony_arrests", "drug_arrests", "dui_arrests",
        "arrest_rate_per_1k", "felony_arrest_rate_per_1k",
        "drug_arrest_rate_per_1k", "dui_arrest_rate_per_1k",
        "all_hours_arrest_to_night_crime_ratio",
    ]),
    ("late_night_citibike_cd.csv", [
        "citibike_trips_night", "citibike_stations",
        "citibike_trips_night_per_1k", "citibike_stations_per_km2",
        "citibike_coverage_flag",
    ]),
    ("night_film_permits_cd.csv", [
        "night_film_permits", "film_activity_per_1k",
    ]),
    ("restaurant_density_cd.csv", [
        "restaurant_count", "late_night_food_count", "cuisine_diversity",
        "restaurants_per_1k", "late_night_food_per_1k",
    ]),
    ("commercial_waste_cd.csv", [
        "n_waste_zones", "commercial_waste_zone_pct",
    ]),

    # ── Tier 3 new data ──
    ("energy_benchmarking_cd.csv", [
        "energy_use_per_sqft", "electricity_use_per_sqft",
        "large_buildings_per_km2",
    ]),
    ("helicopter_noise_cd.csv", [
        "helicopter_complaints_night",
        "helicopter_complaints_night_per_1k", "helicopter_pct_of_noise",
    ]),
    ("subway_frequency_cd.csv", [
        "subway_stations_late_night", "mean_trains_per_hour_late_night",
        "subway_last_train_hour", "late_night_transit_gap",
    ]),
    ("linknyc_wifi_cd.csv", [
        "wifi_hotspots", "linknyc_kiosks",
        "wifi_hotspots_per_km2", "linknyc_kiosks_per_1k",
    ]),
    ("ems_response_cd.csv", [
        "ems_response_min_night", "ems_response_min_day",
        "ems_night_day_response_ratio", "ems_incidents_night_est",
    ]),
]


def load_source(filename: str, columns: list, logger) -> pd.DataFrame:
    """Load a report file and keep only specified columns + boro_cd."""
    path = REPORTS_DIR / filename
    if not path.exists():
        logger.warning(f"  MISSING: {filename}")
        return None

    if filename.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "boro_cd" not in df.columns:
        # Handle borough-level files: expand to CD-level by mapping boro code
        if "boro" in df.columns:
            logger.info(f"  {filename}: borough-level data (boro col) — expanding to CD-level")
            df["boro"] = pd.to_numeric(df["boro"], errors="coerce").astype("Int64")
            cd_boro_map = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
            cd_boro_map["boro"] = cd_boro_map["boro_cd"] // 100
            df = cd_boro_map.merge(df, on="boro", how="left").drop(columns=["boro"])
        else:
            logger.warning(f"  NO boro_cd: {filename}")
            return None

    df["boro_cd"] = pd.to_numeric(df["boro_cd"], errors="coerce").astype("Int64")

    # Drop population/area_km2 from source files to avoid conflicts
    # (these come from the legacy master, the authoritative source)
    leak_cols = {"population", "area_km2"}
    leak_found = leak_cols & (set(df.columns) - set(columns))
    if leak_found:
        df = df.drop(columns=list(leak_found), errors="ignore")

    # Keep only requested columns that exist
    keep = ["boro_cd"]
    missing_cols = []
    for c in columns:
        if c in df.columns:
            keep.append(c)
        else:
            missing_cols.append(c)

    if missing_cols:
        logger.warning(f"  {filename}: missing cols {missing_cols}")

    return df[keep].copy()


def main():
    with get_logger("50_build_master_nightscape") as logger:
        # Start with 59 CDs
        master = pd.DataFrame({"boro_cd": sorted(VALID_CDS)})
        master["boro_cd"] = master["boro_cd"].astype("Int64")

        sources_loaded = 0
        total_vars = 0

        for filename, columns in SOURCES:
            df = load_source(filename, columns, logger)
            if df is None:
                continue

            # Check for column conflicts before merge
            existing_cols = set(master.columns) - {"boro_cd"}
            new_cols = set(df.columns) - {"boro_cd"}
            conflicts = existing_cols & new_cols
            if conflicts:
                # Drop conflicting columns from the new source (keep first occurrence)
                logger.warning(f"  {filename}: dropping duplicate cols {conflicts}")
                df = df.drop(columns=list(conflicts))

            # Validate no duplicate boro_cd in source before merging
            if df["boro_cd"].duplicated().any():
                dup_cds = df.loc[df["boro_cd"].duplicated(keep=False), "boro_cd"].unique().tolist()
                n_dups = df["boro_cd"].duplicated().sum()
                logger.warning(f"  {filename}: {n_dups} duplicate boro_cd rows (CDs: {dup_cds}), keeping first")
                df = df.drop_duplicates(subset=["boro_cd"], keep="first")

            pre_cols = len(master.columns)
            master = master.merge(df, on="boro_cd", how="left")
            assert len(master) == 59, f"Row count changed to {len(master)} after merging {filename}"
            added = len(master.columns) - pre_cols
            sources_loaded += 1
            total_vars += added
            logger.info(f"  {filename}: +{added} cols (total: {len(master.columns) - 1})")

        logger.info(f"\nLoaded {sources_loaded}/{len(SOURCES)} sources")
        logger.info(f"Master shape: {master.shape[0]} rows x {master.shape[1]} cols")

        # Add CD labels
        cd_lookup_path = GEO_DIR / "cd_lookup.csv"
        if cd_lookup_path.exists():
            cd_lookup = pd.read_csv(cd_lookup_path)
            cd_lookup["boro_cd"] = pd.to_numeric(cd_lookup["boro_cd"], errors="coerce").astype("Int64")
            cd_lookup = cd_lookup.drop_duplicates(subset=["boro_cd"])
            # Use cd_short (the actual column name) as cd_name
            if "cd_short" in cd_lookup.columns:
                cd_lookup = cd_lookup.rename(columns={"cd_short": "cd_name"})
            if "cd_name" in cd_lookup.columns:
                master = master.merge(cd_lookup[["boro_cd", "cd_name"]], on="boro_cd", how="left")

        # Validate
        assert len(master) == 59, f"Expected 59 rows, got {len(master)}"
        assert master["boro_cd"].nunique() == 59

        # Check completeness
        numeric_cols = master.select_dtypes(include="number").columns.drop("boro_cd", errors="ignore")
        missing_pct = master[numeric_cols].isna().mean()
        high_missing = missing_pct[missing_pct > 0.20]
        if len(high_missing) > 0:
            logger.warning(f"Columns with >20% missing: {dict(high_missing.round(2))}")
        else:
            logger.info("All numeric columns have ≤20% missing values")

        # Check for negative rates
        rate_cols = [c for c in numeric_cols if "per_1k" in c or "per_km2" in c or "rate" in c]
        for c in rate_cols:
            neg = (master[c] < 0).sum()
            if neg > 0:
                logger.warning(f"  {c}: {neg} negative values")

        master = ensure_boro_cd_dtype(master)
        master = filter_standard_cds(master)

        # Write both parquet and CSV
        atomic_write_df(master, OUTPUT_PATH, index=False)
        atomic_write_df(master, OUTPUT_CSV, index=False)

        logger.log_metrics({
            "sources_loaded": sources_loaded,
            "total_columns": len(master.columns),
            "numeric_columns": len(numeric_cols),
            "rows": len(master),
        })

        print(f"\nMaster Nightscape DataFrame built:")
        print(f"  Shape: {master.shape[0]} rows x {master.shape[1]} columns")
        print(f"  Sources merged: {sources_loaded}/{len(SOURCES)}")
        print(f"  Numeric variables: {len(numeric_cols)}")
        print(f"  Missing values: {master[numeric_cols].isna().sum().sum()} total")
        print(f"  Output: {OUTPUT_PATH}")
        print(f"  CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
