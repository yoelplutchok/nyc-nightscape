"""
01_migrate_data.py — Migrate data from NYC Night Signals to NYC Nightscape.

Copies raw data caches, processed CD-level reports, equity/index files,
and legacy config. Validates the migration afterward.
"""

import shutil
import sys
from pathlib import Path

# ── Source and target ──────────────────────────────────────────────────────────
SOURCE = Path("/Users/yoelplutchok/Desktop/Sleep_ESI_NYC")
TARGET = Path(__file__).resolve().parent.parent

RAW_SRC = SOURCE / "data" / "raw"
RAW_DST = TARGET / "data" / "raw"

REPORTS_SRC = SOURCE / "data" / "processed" / "reports"
REPORTS_DST = TARGET / "data" / "processed" / "reports"

EQUITY_SRC = SOURCE / "data" / "processed" / "equity"
EQUITY_DST = TARGET / "data" / "processed" / "equity"

# ── What to copy ──────────────────────────────────────────────────────────────

RAW_DIRS_TO_COPY = [
    "311_noise",           # 1.5 GB
    "nypd_shots",          # 8.3 MB
    "nypd_crime",          # 128 KB
    "ahv_permits",         # 60 MB
    "dob_permits",         # 40 MB
    "dot_traffic",         # 32 MB
    "ecb_noise",           # 4.3 MB
    "nycha",               # 660 KB
    "dep_noise",           # 98 MB
    "pluto",               # 16 KB
    "sla",                 # 3.7 MB
    "noise_objective",     # 4.2 GB
    "light_viirs",         # 483 MB
    "air_nyccas",          # 17 MB
    "heat_prism",          # 1.2 GB
    "acs_demographics",    # 592 KB
    "acs_population",      # 120 KB
    "cdc_places",          # 72 KB
    "chs",                 # 1.8 MB
    "sonyc",               # 14 MB
    "community_districts", # 3.6 MB
    "census_tracts_2020",  # 13 MB
    "ntas_2020",           # 13 MB
    "uhf42",               # 168 KB
    "dot_noise",           # 11 MB
]

REPORTS_TO_COPY = [
    "master_analysis_df.parquet",
    "shots_fired_cd.csv",
    "siren_density_cd.csv",
    "nighttime_crime_cd.csv",
    "ahv_permits_cd.csv",
    "dob_construction_density_cd.csv",
    "dot_traffic_cd.csv",
    "ecb_noise_violations_cd.csv",
    "nycha_noise_exposure.csv",
    "cleaned_311_rates_cd.csv",
    "response_time_cd.csv",
    "resolution_equity_cd.csv",
    "descriptor_rates_cd.csv",
    "pluto_cd_metrics.csv",
    "liquor_license_cd.csv",
    "dep_noise_summary.csv",
    "chs_cd_estimates.csv",
    "population_at_risk.csv",
    "borough_summary_table.csv",
    "descriptive_statistics_table1.csv",
]

EQUITY_TO_COPY = [
    "esi_with_acs.parquet",
    "ej_index_cd.parquet",
    "dual_reality_cd.parquet",
    "noise_inequality_indices.csv",
]


def dir_size_bytes(path: Path) -> int:
    """Recursively compute total size of a directory."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def copy_raw_dirs():
    """Copy raw data cache directories."""
    print("\n=== Step 1: Copy raw data caches ===")
    copied = 0
    skipped = 0
    missing = []

    for dirname in RAW_DIRS_TO_COPY:
        src = RAW_SRC / dirname
        dst = RAW_DST / dirname

        if not src.exists():
            print(f"  MISSING: {dirname}")
            missing.append(dirname)
            continue

        if dst.exists():
            print(f"  SKIP (exists): {dirname}")
            skipped += 1
            continue

        size_mb = dir_size_bytes(src) / 1_000_000
        print(f"  Copying {dirname} ({size_mb:.1f} MB)...")
        shutil.copytree(src, dst)
        copied += 1

    print(f"\n  Copied: {copied}, Skipped: {skipped}, Missing: {len(missing)}")
    if missing:
        print(f"  Missing dirs: {missing}")
    return missing


def copy_reports():
    """Copy processed CD-level report files."""
    print("\n=== Step 2: Copy processed CD-level reports ===")
    REPORTS_DST.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = []

    for fname in REPORTS_TO_COPY:
        src = REPORTS_SRC / fname
        dst = REPORTS_DST / fname

        if not src.exists():
            print(f"  MISSING: {fname}")
            missing.append(fname)
            continue

        if dst.exists():
            print(f"  SKIP (exists): {fname}")
            continue

        shutil.copy2(src, dst)
        print(f"  Copied: {fname}")
        copied += 1

    print(f"\n  Copied: {copied}, Missing: {len(missing)}")
    if missing:
        print(f"  Missing files: {missing}")
    return missing


def copy_equity():
    """Copy equity/index files."""
    print("\n=== Step 3: Copy equity/index files ===")
    EQUITY_DST.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = []

    for fname in EQUITY_TO_COPY:
        src = EQUITY_SRC / fname
        dst = EQUITY_DST / fname

        if not src.exists():
            print(f"  MISSING: {fname}")
            missing.append(fname)
            continue

        if dst.exists():
            print(f"  SKIP (exists): {fname}")
            continue

        shutil.copy2(src, dst)
        print(f"  Copied: {fname}")
        copied += 1

    print(f"\n  Copied: {copied}, Missing: {len(missing)}")
    if missing:
        print(f"  Missing files: {missing}")
    return missing


def copy_legacy_config():
    """Copy legacy params.yml for reference."""
    print("\n=== Step 4: Copy legacy params.yml ===")
    src = SOURCE / "configs" / "params.yml"
    dst = TARGET / "configs" / "params_legacy.yml"

    if dst.exists():
        print("  SKIP (exists): params_legacy.yml")
        return

    if not src.exists():
        print("  MISSING: source configs/params.yml")
        return

    shutil.copy2(src, dst)
    print("  Copied: configs/params_legacy.yml")


def validate():
    """Run migration validation checks."""
    print("\n=== Step 5: Validation ===")
    import pandas as pd

    errors = []

    # Check master_analysis_df
    master_path = REPORTS_DST / "master_analysis_df.parquet"
    if master_path.exists():
        df = pd.read_parquet(master_path)
        print(f"  master_analysis_df: {len(df)} rows, {len(df.columns)} columns")
        if len(df) != 59:
            errors.append(f"master_analysis_df has {len(df)} rows, expected 59")
        else:
            print("  OK: 59 CDs present")

        if "boro_cd" in df.columns:
            na_count = df["boro_cd"].isna().sum()
            if na_count > 0:
                errors.append(f"boro_cd has {na_count} NaN values")
            else:
                print("  OK: No NaN in boro_cd")
    else:
        errors.append("master_analysis_df.parquet not found")

    # Check CD-level CSVs have boro_cd with valid values
    valid_cds = set()
    for boro, max_cd in [(1, 12), (2, 12), (3, 18), (4, 14), (5, 3)]:
        for cd in range(1, max_cd + 1):
            valid_cds.add(boro * 100 + cd)

    csv_files = [f for f in REPORTS_TO_COPY if f.endswith(".csv")]
    checked = 0
    for fname in csv_files:
        fpath = REPORTS_DST / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        if "boro_cd" in df.columns:
            cds = set(df["boro_cd"].dropna().astype(int))
            invalid = cds - valid_cds
            if invalid:
                errors.append(f"{fname}: invalid boro_cd values: {invalid}")
            checked += 1

    print(f"  Checked {checked} CSV files for valid boro_cd values")

    # Check cd59.geojson
    geo_path = TARGET / "data" / "processed" / "geo" / "cd59.geojson"
    if geo_path.exists():
        import geopandas as gpd
        gdf = gpd.read_file(geo_path)
        print(f"  cd59.geojson: {len(gdf)} features")
        if len(gdf) != 59:
            errors.append(f"cd59.geojson has {len(gdf)} features, expected 59")
        else:
            print("  OK: 59 CD boundaries")
    else:
        errors.append("cd59.geojson not found in data/processed/geo/")

    # Check raw directories exist
    raw_present = sum(1 for d in RAW_DIRS_TO_COPY if (RAW_DST / d).exists())
    print(f"  Raw directories: {raw_present}/{len(RAW_DIRS_TO_COPY)} present")
    if raw_present < len(RAW_DIRS_TO_COPY):
        missing = [d for d in RAW_DIRS_TO_COPY if not (RAW_DST / d).exists()]
        errors.append(f"Missing raw directories: {missing}")

    # Check total raw data size
    total_raw_bytes = dir_size_bytes(RAW_DST)
    total_raw_gb = total_raw_bytes / 1_000_000_000
    print(f"  Total raw data: {total_raw_gb:.2f} GB")
    if total_raw_gb < 5.0:
        errors.append(f"Raw data seems too small: {total_raw_gb:.2f} GB (expected 6-7 GB)")

    # Check for zero-size files
    zero_files = []
    for f in RAW_DST.rglob("*"):
        if f.is_file() and f.stat().st_size == 0:
            zero_files.append(str(f.relative_to(RAW_DST)))
    if zero_files:
        errors.append(f"{len(zero_files)} zero-size files found: {zero_files[:5]}")
        print(f"  WARNING: {len(zero_files)} zero-size files")
    else:
        print("  OK: No zero-size files")

    # Summary
    print("\n=== MIGRATION SUMMARY ===")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("ALL CHECKS PASSED")
        return True


def main():
    print("=" * 60)
    print("NYC Nightscape — Data Migration from Night Signals")
    print("=" * 60)
    print(f"Source: {SOURCE}")
    print(f"Target: {TARGET}")

    if not SOURCE.exists():
        print(f"\nERROR: Source project not found at {SOURCE}")
        sys.exit(1)

    # Step 1: Raw data
    raw_missing = copy_raw_dirs()

    # Step 2: Reports
    report_missing = copy_reports()

    # Step 3: Equity
    equity_missing = copy_equity()

    # Step 4: Legacy config
    copy_legacy_config()

    # Step 5: Validate
    ok = validate()

    if not ok:
        print("\nMigration completed with errors. Review above.")
        sys.exit(1)
    else:
        print("\nMigration completed successfully!")


if __name__ == "__main__":
    main()
